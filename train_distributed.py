import argparse
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from displayUtils import DisplayUtils
from hands_dataset import get_dataset_root, load_combined_metadata, set_dataset_root
from metrics import LossWeights, weighted_regression_loss
from models import EFFICIENTNET_IMG_SIZES, build_age_model
from train_age import (
    AgeDataset,
    build_transforms,
    compute_age_gate_curves,
    filter_metadata,
    multimodal_collate,
    stratified_user_split,
    set_random_seed,
)

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_LR = 3e-4
DEFAULT_SEED = 42
DEFAULT_MIN_DELTA = 0.001
DEFAULT_PATIENCE = 20
DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_POINTS = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed training for multimodal age regressor (RGB / PointCloud).")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--no-stratified-user-split",
        action="store_true",
        help="Disable per-user stratification when splitting the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where checkpoints and plots will be saved.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Input resolution for RGB branch (default: {DEFAULT_IMG_SIZE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Per-process mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: 40).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience in epochs (default: {DEFAULT_PATIENCE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for RNGs (default: 42).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for AdamW optimizer (default: 3e-4).",
    )
    parser.add_argument(
        "--loss-weight-nll",
        type=float,
        default=0.5,
        help="Weight for the Gaussian NLL component (default: 0.5; set to 1.0 for legacy behaviour).",
    )
    parser.add_argument(
        "--loss-weight-mse",
        type=float,
        default=0.25,
        help="Weight for the MSE component (default: 0.25).",
    )
    parser.add_argument(
        "--loss-weight-mae",
        type=float,
        default=0.25,
        help="Weight for the MAE component (default: 0.25).",
    )
    parser.add_argument(
        "--no-rgb",
        action="store_true",
        help="Disable RGB branch (point cloud only).",
    )
    parser.add_argument(
        "--use-pointcloud",
        action="store_true",
        help="Enable point cloud branch (requires xyz_npy files).",
    )
    parser.add_argument(
        "--rgb-backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"] + [f"efficientnet_{k}" for k in EFFICIENTNET_IMG_SIZES.keys()],
        help="RGB backbone for the CNN encoder (default: resnet18).",
    )
    parser.add_argument(
        "--rgb-latent-dim",
        type=int,
        default=256,
        help="Latent dimensionality for the RGB encoder (default: 256).",
    )
    parser.add_argument(
        "--pc-latent-dim",
        type=int,
        default=256,
        help="Latent dimensionality for the point cloud encoder (default: 256).",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=256,
        help="Hidden size of the fusion MLP head (default: 256).",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.1,
        help="Dropout for the fusion head (default: 0.1).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help=f"Number of points to sample from each point cloud (default: {DEFAULT_NUM_POINTS}).",
    )
    parser.add_argument(
        "--pc-jitter-std",
        type=float,
        default=0.0,
        help="Gaussian jitter stddev applied to point clouds (default: 0.0).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers per process (default: 2).",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="torch.distributed backend to use.",
    )
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=1800,
        help="Timeout (in seconds) for torch.distributed initialization.",
    )
    parser.add_argument(
        "--find-unused-params",
        action="store_true",
        help="Enable DistributedDataParallel(find_unused_parameters=True).",
    )
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    timeout = timedelta(seconds=int(args.dist_timeout))
    dist.init_process_group(backend=args.dist_backend, timeout=timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, device


def build_datasets(args: argparse.Namespace, seed: int, use_rgb: bool, use_pointcloud: bool):
    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()

    train_transform = test_transform = None
    if use_rgb:
        train_transform, test_transform = build_transforms(int(args.img_size))

    metadata = filter_metadata(
        load_combined_metadata(root=active_root),
        require_xyz=use_pointcloud,
    )
    if args.no_stratified_user_split:
        user_ids = metadata["user_id"].unique()
        train_ids, val_ids = train_test_split(user_ids, test_size=0.2, random_state=seed)
    else:
        train_ids, val_ids = stratified_user_split(
            metadata,
            test_size=0.2,
            random_state=seed,
        )
    train_meta = metadata[metadata["user_id"].isin(train_ids)]
    val_meta = metadata[metadata["user_id"].isin(val_ids)]

    if use_pointcloud and (train_meta.empty or val_meta.empty):
        raise ValueError("Point cloud branch enabled but dataset split is empty; check xyz_npy availability.")

    train_ds = AgeDataset(
        train_meta,
        use_rgb=use_rgb,
        use_pointcloud=use_pointcloud,
        transform=train_transform,
        num_points=args.num_points,
        pc_jitter_std=args.pc_jitter_std,
    )
    val_ds = AgeDataset(
        val_meta,
        use_rgb=use_rgb,
        use_pointcloud=use_pointcloud,
        transform=test_transform,
        num_points=args.num_points,
        pc_jitter_std=0.0,
    )
    return train_ds, val_ds, active_root, len(train_meta), len(val_meta)


def build_dataloaders(
    train_dataset,
    val_dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    world_size: int,
    rank: int,
):
    pin_memory = device.type == "cuda"
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=multimodal_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=multimodal_collate,
    )
    return train_loader, val_loader, train_sampler


def all_reduce_metrics(device: torch.device, sums: list[float]) -> list[float]:
    tensor = torch.tensor(sums, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def gather_all_lists(local_list, world_size: int):
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, list(local_list))
    merged = []
    for part in gather_list:
        if part:
            merged.extend(part)
    return merged


def main() -> None:
    args = parse_args()
    loss_weights = LossWeights(
        nll=args.loss_weight_nll,
        mse=args.loss_weight_mse,
        mae=args.loss_weight_mae,
    )
    loss_weights.validate()
    rank, world_size, local_rank, device = init_distributed(args)
    is_main = rank == 0

    use_rgb = not args.no_rgb
    use_pointcloud = args.use_pointcloud
    if not use_rgb and not use_pointcloud:
        raise ValueError("At least one modality must be enabled (RGB and/or point cloud).")

    set_random_seed(args.seed + rank)

    train_dataset, val_dataset, active_root, train_len, val_len = build_datasets(
        args,
        args.seed,
        use_rgb=use_rgb,
        use_pointcloud=use_pointcloud,
    )
    train_loader, val_loader, train_sampler = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        world_size=world_size,
        rank=rank,
    )

    output_dir = Path(args.output_dir).expanduser()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Using dataset root: {active_root}\n"
            f"Saving artifacts to: {output_dir}\n"
            f"Train images: {train_len}\n"
            f"Val images:   {val_len}\n"
            f"Modalities: {'RGB' if use_rgb else ''}{' + ' if use_rgb and use_pointcloud else ''}{'PointCloud' if use_pointcloud else ''} | "
            f"Image size: {args.img_size} | Points: {args.num_points} | "
            f"Per-rank batch size: {args.batch_size}\n"
            f"Epochs: {args.epochs} | Learning rate: {args.lr:.2e} | Seed: {args.seed} | World size: {world_size}"
        )
        print(
            f"Loss weights -> NLL: {loss_weights.nll:.3f}, "
            f"MSE: {loss_weights.mse:.3f}, MAE: {loss_weights.mae:.3f}"
        )
        split_desc = (
            "Unstratified per-user split (random)."
            if args.no_stratified_user_split
            else "Stratified per-user split (adult/minor aware)."
        )
        print(f"Split mode: {split_desc}")

    model = build_age_model(
        use_rgb=use_rgb,
        use_point_cloud=use_pointcloud,
        rgb_backbone=args.rgb_backbone,
        rgb_latent_dim=args.rgb_latent_dim,
        pc_latent_dim=args.pc_latent_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    ).to(device)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        find_unused_parameters=args.find_unused_params,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)

    modalities = []
    if use_rgb:
        modalities.append("rgb")
    if use_pointcloud:
        modalities.append("pc")
    model_tag = "+".join(modalities) if modalities else "unknown"

    best_val_loss = float("inf")
    best_model_path = output_dir / f"{model_tag}_age_regressor_ddp.pth"
    history_log_path = output_dir / "history_distributed.log"
    min_delta = DEFAULT_MIN_DELTA
    patience = max(1, int(args.patience))
    epochs_without_improvement = 0
    history_entries = [] if is_main else None

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        train_sample_count = 0.0
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_mse_sum = 0.0
        train_std_sum = 0.0

        progress = tqdm(
            train_loader,
            desc=f"[Rank {rank}] Epoch {epoch}/{args.epochs}",
            disable=not is_main,
        )
        for images, points, ages in progress:
            if images is not None:
                images = images.to(device, non_blocking=True)
            if points is not None:
                points = points.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred_mean, pred_log_var = ddp_model(images=images, points=points)
            loss = weighted_regression_loss(pred_mean, pred_log_var, ages, loss_weights)
            loss.backward()
            optimizer.step()

            batch_size = ages.size(0)
            train_sample_count += batch_size
            train_loss_sum += loss.item() * batch_size
            abs_err = torch.abs(pred_mean - ages)
            sq_err = (pred_mean - ages) ** 2
            batch_std = torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))
            train_mae_sum += torch.sum(abs_err).item()
            train_mse_sum += torch.sum(sq_err).item()
            train_std_sum += torch.sum(batch_std).item()

        train_totals = all_reduce_metrics(
            device,
            [train_loss_sum, train_mae_sum, train_mse_sum, train_std_sum, train_sample_count],
        )
        train_loss = train_totals[0] / max(1.0, train_totals[4])
        train_mae = train_totals[1] / max(1.0, train_totals[4])
        train_rmse = float(np.sqrt(train_totals[2] / max(1.0, train_totals[4])))
        train_std = train_totals[3] / max(1.0, train_totals[4])

        ddp_model.eval()
        val_sample_count = 0.0
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_mse_sum = 0.0
        val_std_sum = 0.0
        val_targets = []
        val_predictions = []
        val_log_vars = []

        with torch.no_grad():
            for images, points, ages in val_loader:
                if images is not None:
                    images = images.to(device, non_blocking=True)
                if points is not None:
                    points = points.to(device, non_blocking=True)
                ages = ages.to(device, non_blocking=True)
                pred_mean, pred_log_var = ddp_model(images=images, points=points)
                batch_loss = weighted_regression_loss(pred_mean, pred_log_var, ages, loss_weights)

                batch_size = ages.size(0)
                val_sample_count += batch_size
                val_loss_sum += batch_loss.item() * batch_size
                val_mae_sum += torch.sum(torch.abs(pred_mean - ages)).item()
                val_mse_sum += torch.sum((pred_mean - ages) ** 2).item()
                val_std_sum += torch.sum(
                    torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))
                ).item()

                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(pred_mean.detach().cpu().tolist())
                val_log_vars.extend(pred_log_var.detach().cpu().tolist())

        val_totals = all_reduce_metrics(
            device,
            [val_loss_sum, val_mae_sum, val_mse_sum, val_std_sum, val_sample_count],
        )
        val_loss = val_totals[0] / max(1.0, val_totals[4])
        val_mae = val_totals[1] / max(1.0, val_totals[4])
        val_rmse = float(np.sqrt(val_totals[2] / max(1.0, val_totals[4])))
        val_std = val_totals[3] / max(1.0, val_totals[4])

        if is_main:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, train_rmse={train_rmse:.4f}, train_std={train_std:.4f} | "
                f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, val_rmse={val_rmse:.4f}, val_std={val_std:.4f}"
            )
            with history_log_path.open("a", encoding="utf-8") as log_fp:
                log_fp.write(
                    f"Epoch {epoch},train_loss={train_loss:.6f},train_mae={train_mae:.6f},train_rmse={train_rmse:.6f},"
                    f"train_std={train_std:.6f},val_loss={val_loss:.6f},val_mae={val_mae:.6f},val_rmse={val_rmse:.6f},val_std={val_std:.6f}\n"
                )
            history_entries.append(
                {
                    "epoch": epoch,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                }
            )

        previous_best = best_val_loss
        save_best = val_loss < best_val_loss
        if save_best:
            improvement = (
                float("inf") if previous_best == float("inf") else previous_best - val_loss
            )
            best_val_loss = val_loss
            if improvement == float("inf") or improvement >= min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if save_best:
            targets_all = gather_all_lists(val_targets, world_size)
            preds_all = gather_all_lists(val_predictions, world_size)
            log_vars_all = gather_all_lists(val_log_vars, world_size)
        else:
            targets_all = preds_all = log_vars_all = None

        if save_best and is_main:
            model_to_save = ddp_model.module
            torch.save(model_to_save.state_dict(), best_model_path)

            plot_path = output_dir / f"age_val_scatter_epoch{epoch}_ddp.png"
            if DisplayUtils.save_regression_scatter(
                targets_all,
                preds_all,
                save_path=plot_path,
                title=f"Epoch {epoch} Age Predictions (best so far, DDP)",
                axis_limits=(0.0, 70.0),
                point_size=20,
                alpha=0.6,
            ):
                print(
                    f"[Rank 0] Saved best model to {best_model_path} (val_loss={val_loss:.4f}) "
                    f"and plot to {plot_path}"
                )

            gate_results = compute_age_gate_curves(
                np.asarray(targets_all, dtype=float),
                np.asarray(preds_all, dtype=float),
                np.asarray(log_vars_all, dtype=float),
                age_threshold=18.0,
                num_thresholds=201,
            )

            preds_dump_path = output_dir / "best_val_predictions_ddp.npz"
            np.savez(
                preds_dump_path,
                targets=np.asarray(targets_all, dtype=float),
                pred_mean=np.asarray(preds_all, dtype=float),
                pred_log_var=np.asarray(log_vars_all, dtype=float),
                adult_prob=gate_results["adult_prob"],
                epoch=epoch,
            )

            roc_case1_path = output_dir / "roc_case1_adult_gate_ddp.png"
            roc_case2_path = output_dir / "roc_case2_child_gate_ddp.png"
            DisplayUtils.plot_roc_curve(
                gate_results["case1"]["fpr"],
                gate_results["case1"]["tpr"],
                thresholds=gate_results["case1"]["thresholds"],
                save_path=roc_case1_path,
                title="ROC - Adult Content Gate (admit adults, DDP)",
                auc_value=gate_results["case1"]["auc"],
                show=False,
            )
            DisplayUtils.plot_roc_curve(
                gate_results["case2"]["fpr"],
                gate_results["case2"]["tpr"],
                thresholds=gate_results["case2"]["thresholds"],
                save_path=roc_case2_path,
                title="ROC - Child Platform Gate (admit minors, DDP)",
                auc_value=gate_results["case2"]["auc"],
                show=False,
            )

            metrics_csv_path = output_dir / "age_gate_metrics_ddp.csv"
            with metrics_csv_path.open("w", encoding="utf-8") as metrics_fp:
                metrics_fp.write("case,tau,fpr,fnr,tpr,tnr\n")
                for case_name, case_data in (
                    ("adult_content_gate", gate_results["case1"]),
                    ("child_platform_gate", gate_results["case2"]),
                ):
                    for tau, fpr, fnr, tpr_val, tnr in zip(
                        case_data["thresholds"],
                        case_data["fpr"],
                        case_data["fnr"],
                        case_data["tpr"],
                        case_data["tnr"],
                    ):
                        metrics_fp.write(
                            f"{case_name},{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n"
                        )

            print(
                f"[Rank 0] Updated ROC plots ({roc_case1_path.name}, {roc_case2_path.name}), "
                f"metrics CSV ({metrics_csv_path.name}), and predictions dump ({preds_dump_path.name})."
            )

        if epochs_without_improvement >= patience:
            if is_main:
                print(
                    f"Stopping early at epoch {epoch}: validation loss did not improve by at least "
                    f"{min_delta:.3f} for {patience} consecutive epochs."
                )
                history_plot_path = output_dir / "history_plot_ddp.png"
                DisplayUtils.plot_loss_history(
                    history_entries,
                    save_path=history_plot_path,
                    show=False,
                    title="Training History (MAE & RMSE, DDP)",
                )
                print(f"[Rank 0] Saved training history plot to {history_plot_path}")
            break

    if is_main:
        print("Distributed training complete. Best model saved based on validation improvement.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
