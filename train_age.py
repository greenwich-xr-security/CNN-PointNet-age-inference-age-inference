import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.age import AgeDataset, DEFAULT_NUM_POINTS
from datasets.hands_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from datasets.transforms import build_transforms
from datasets.utils import filter_metadata, multimodal_collate, stratified_user_split
from displayUtils import DisplayUtils
from metrics import LossWeights, weighted_regression_loss
from models import EFFICIENTNET_IMG_SIZES, FusionConfig, build_age_model

# Example (Windows): python train_age.py --data-root "C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets" --output-dir runs\\multimodal --img-size 224 --batch-size 32 --epochs 40 --seed 42 --lr 0.0003 --use-pointcloud

# --- Config -----------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_LR = 3e-4
DEFAULT_SEED = 42
DEFAULT_IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_PATIENCE = 20

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def compute_adult_probabilities(
    pred_means,
    pred_log_vars,
    *,
    age_threshold: float = 18.0,
) -> np.ndarray:
    """Return P(age >= threshold) from predicted Gaussian parameters."""
    means = torch.as_tensor(pred_means, dtype=torch.float32, device="cpu")
    log_vars = torch.as_tensor(pred_log_vars, dtype=torch.float32, device="cpu")
    log_vars = torch.clamp(log_vars, min=-10.0, max=10.0)
    std = torch.exp(0.5 * log_vars)
    std = torch.clamp(std, min=1e-3)
    z = (age_threshold - means) / std
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    adult_prob = torch.clamp(1.0 - cdf, min=0.0, max=1.0)
    return adult_prob.numpy()


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_age_gate_curves(
    targets,
    pred_means,
    pred_log_vars,
    *,
    age_threshold: float = 18.0,
    num_thresholds: int = 101,
) -> dict:
    """Compute ROC-style metrics for both policy cases using adult probabilities."""
    targets_arr = np.asarray(targets, dtype=float)
    adult_prob = compute_adult_probabilities(pred_means, pred_log_vars, age_threshold=age_threshold)
    tau_values = np.linspace(0.0, 1.0, num=num_thresholds)

    is_adult = targets_arr >= age_threshold
    is_minor = ~is_adult
    adult_total = int(is_adult.sum())
    minor_total = int(is_minor.sum())

    def build_case(admit_mask, positive_mask, negative_mask):
        tp = np.logical_and(admit_mask, positive_mask).sum()
        fp = np.logical_and(admit_mask, negative_mask).sum()
        fn = np.logical_and(~admit_mask, positive_mask).sum()
        tn = np.logical_and(~admit_mask, negative_mask).sum()
        pos_total = positive_mask.sum()
        neg_total = negative_mask.sum()
        tpr = _safe_rate(tp, pos_total)
        fpr = _safe_rate(fp, neg_total)
        fnr = _safe_rate(fn, pos_total)
        tnr = _safe_rate(tn, neg_total)
        return fpr, tpr, fnr, tnr

    case1_fprs = []
    case1_tprs = []
    case1_fnrs = []
    case1_tnrs = []
    case2_fprs = []
    case2_tprs = []
    case2_fnrs = []
    case2_tnrs = []

    for tau in tau_values:
        admit_adult = adult_prob >= tau  # Case 1
        fpr1, tpr1, fnr1, tnr1 = build_case(admit_adult, is_adult, is_minor)
        case1_fprs.append(fpr1)
        case1_tprs.append(tpr1)
        case1_fnrs.append(fnr1)
        case1_tnrs.append(tnr1)

        admit_minor = adult_prob < tau  # Case 2
        fpr2, tpr2, fnr2, tnr2 = build_case(admit_minor, is_minor, is_adult)
        case2_fprs.append(fpr2)
        case2_tprs.append(tpr2)
        case2_fnrs.append(fnr2)
        case2_tnrs.append(tnr2)

    def compute_auc(fprs, tprs):
        fprs_arr = np.asarray(fprs, dtype=float)
        tprs_arr = np.asarray(tprs, dtype=float)
        order = np.argsort(fprs_arr)
        if fprs_arr.size == 0:
            return 0.0
        return float(np.trapz(tprs_arr[order], fprs_arr[order]))

    results = {
        "adult_prob": adult_prob,
        "case1": {
            "fpr": np.asarray(case1_fprs, dtype=float),
            "tpr": np.asarray(case1_tprs, dtype=float),
            "fnr": np.asarray(case1_fnrs, dtype=float),
            "tnr": np.asarray(case1_tnrs, dtype=float),
            "thresholds": tau_values,
            "auc": compute_auc(case1_fprs, case1_tprs),
            "adult_total": adult_total,
            "minor_total": minor_total,
        },
        "case2": {
            "fpr": np.asarray(case2_fprs, dtype=float),
            "tpr": np.asarray(case2_tprs, dtype=float),
            "fnr": np.asarray(case2_fnrs, dtype=float),
            "tnr": np.asarray(case2_tnrs, dtype=float),
            "thresholds": tau_values,
            "auc": compute_auc(case2_fprs, case2_tprs),
            "adult_total": adult_total,
            "minor_total": minor_total,
        },
    }
    return results


def _confusion_counts(adult_prob: np.ndarray, targets: np.ndarray, tau: float, *, age_threshold: float = 18.0):
    """Return TP, TN, FP, FN counts for case1 (admit adults) at a given tau."""
    is_adult = targets >= age_threshold
    admit_adult = adult_prob >= tau
    tp = int(np.logical_and(admit_adult, is_adult).sum())
    fp = int(np.logical_and(admit_adult, ~is_adult).sum())
    fn = int(np.logical_and(~admit_adult, is_adult).sum())
    tn = int(np.logical_and(~admit_adult, ~is_adult).sum())
    return tp, tn, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal hand age regressor (RGB / PointCloud).")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
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
        help="Mini-batch size for training (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for RNGs, ensuring reproducibility (default: 42).",
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
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience in epochs (default: {DEFAULT_PATIENCE}).",
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
    args = parser.parse_args()

    loss_weights = LossWeights(
        nll=args.loss_weight_nll,
        mse=args.loss_weight_mse,
        mae=args.loss_weight_mae,
    )
    loss_weights.validate()

    set_random_seed(args.seed)
    use_rgb = not args.no_rgb
    use_pointcloud = args.use_pointcloud
    if not use_rgb and not use_pointcloud:
        raise ValueError("At least one modality must be enabled (RGB and/or point cloud).")

    img_size = int(args.img_size)

    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = test_transform = None
    if use_rgb:
        train_transform, test_transform = build_transforms(img_size)

    metadata = filter_metadata(
        load_combined_metadata(root=active_root),
        require_xyz=use_pointcloud,
    )
    train_ids, test_ids, bin_info = stratified_user_split(
        metadata,
        test_size=0.2,
        random_state=args.seed,
        num_bins=13,
        target_bin_size=30,
        return_bin_info=True,
    )
    train_meta = metadata[metadata["user_id"].isin(train_ids)]
    test_meta = metadata[metadata["user_id"].isin(test_ids)]

    if use_pointcloud and (train_meta.empty or test_meta.empty):
        raise ValueError("Point cloud branch enabled but dataset split is empty; check xyz_npy availability.")

    print(
        f"Using dataset root: {active_root}\n"
        f"Saving artifacts to: {output_dir}\n"
        f"Train users: {train_meta['user_id'].nunique()} | Train images: {len(train_meta)}\n"
        f"Test users:  {test_meta['user_id'].nunique()} | Test images:  {len(test_meta)}\n"
        f"Modalities: {'RGB' if use_rgb else ''}{' + ' if use_rgb and use_pointcloud else ''}{'PointCloud' if use_pointcloud else ''} | "
        f"Image size: {img_size} | Points: {args.num_points} | Batch size: {args.batch_size}\n"
        f"Epochs: {args.epochs} | Learning rate: {args.lr:.2e} | Seed: {args.seed}"
    )
    print(
        f"Loss weights -> NLL: {loss_weights.nll:.3f}, "
        f"MSE: {loss_weights.mse:.3f}, MAE: {loss_weights.mae:.3f}"
    )
    print("Split mode: Stratified per-user split (integer age bins).")

    # Persist split/bins summary for reproducibility.
    split_summary_path = output_dir / "split_summary.txt"
    try:
        with split_summary_path.open("w", encoding="utf-8") as fp:
            fp.write(f"Seed: {args.seed}\n")
            fp.write(f"Users total: {metadata['user_id'].nunique()}\n")
            fp.write(f"Train users: {len(train_ids)} | Test users: {len(test_ids)}\n")
            fp.write("Bins (integer ages):\n")
            if bin_info:
                for b in bin_info:
                    obs_min = b.get("age_min_obs")
                    obs_max = b.get("age_max_obs")
                    obs_span = f"{int(obs_min)}-{int(obs_max)}" if obs_min is not None and obs_max is not None else "n/a"
                    fp.write(
                        f"  Bin {b['bin']}: {int(b['age_min'])}-{int(b['age_max'])} (obs {obs_span}) "
                        f"| users={b['users']} train={b['train_users']} test={b['test_users']}\n"
                    )
            else:
                fp.write("  (no bin info available)\n")
            fp.write("\nTrain user_ids:\n")
            fp.write(", ".join(sorted([str(uid) for uid in train_ids])) + "\n")
            fp.write("\nTest user_ids:\n")
            fp.write(", ".join(sorted([str(uid) for uid in test_ids])) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to write split summary to {split_summary_path}: {exc}")

    train_ds = AgeDataset(
        train_meta,
        use_rgb=use_rgb,
        use_pointcloud=use_pointcloud,
        transform=train_transform,
        num_points=args.num_points,
        pc_jitter_std=args.pc_jitter_std,
    )
    test_ds = AgeDataset(
        test_meta,
        use_rgb=use_rgb,
        use_pointcloud=use_pointcloud,
        transform=test_transform,
        num_points=args.num_points,
        pc_jitter_std=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multimodal_collate,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate,
    )

    model = build_age_model(
        use_rgb=use_rgb,
        use_point_cloud=use_pointcloud,
        rgb_backbone=args.rgb_backbone,
        rgb_latent_dim=args.rgb_latent_dim,
        pc_latent_dim=args.pc_latent_dim,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    )
    if DEVICE.type == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Using {gpu_count} GPUs via DataParallel.")
            model = nn.DataParallel(model)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    modalities = []
    if use_rgb:
        modalities.append("rgb")
    if use_pointcloud:
        modalities.append("pc")
    model_tag = "+".join(modalities) if modalities else "unknown"

    best_val_loss = float("inf")
    best_model_path = output_dir / f"{model_tag}_age_regressor.pth"
    history_log_path = output_dir / "history.log"
    min_delta = 0.001
    patience = max(1, int(args.patience))
    epochs_without_improvement = 0
    history_entries: list[dict] = []
    last_val_user_ids = None
    last_val_targets = None
    last_val_predictions = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        running_std = 0.0
        for images, points, ages, _user_ids in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            if images is not None:
                images = images.to(DEVICE)
            if points is not None:
                points = points.to(DEVICE)
            ages = ages.to(DEVICE)
            optimizer.zero_grad()
            pred_mean, pred_log_var = model(images=images, points=points)
            loss = weighted_regression_loss(pred_mean, pred_log_var, ages, loss_weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            mae = torch.mean(torch.abs(pred_mean - ages)).item()
            mse = torch.mean((pred_mean - ages) ** 2).item()
            avg_std = torch.mean(torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))).item()
            running_mae += mae
            running_mse += mse
            running_std += avg_std

        denom = max(1, len(train_loader))
        train_loss = running_loss / denom
        train_mae = running_mae / denom
        train_mse = running_mse / denom
        train_std = running_std / denom

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        val_std = 0.0
        val_targets = []
        val_predictions = []
        val_log_vars = []
        val_user_ids = []
        with torch.no_grad():
            for images, points, ages, user_ids in test_loader:
                if images is not None:
                    images = images.to(DEVICE)
                if points is not None:
                    points = points.to(DEVICE)
                ages = ages.to(DEVICE)
                pred_mean, pred_log_var = model(images=images, points=points)
                batch_loss = weighted_regression_loss(pred_mean, pred_log_var, ages, loss_weights).item()
                val_loss += batch_loss
                val_mae += torch.mean(torch.abs(pred_mean - ages)).item()
                val_mse += torch.mean((pred_mean - ages) ** 2).item()
                val_std += torch.mean(torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))).item()
                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(pred_mean.detach().cpu().tolist())
                val_log_vars.extend(pred_log_var.detach().cpu().tolist())
                val_user_ids.extend(user_ids)

        denom = max(1, len(test_loader))
        val_loss /= denom
        val_mae /= denom
        val_mse /= denom
        val_std /= denom
        last_val_user_ids = val_user_ids
        last_val_targets = val_targets
        last_val_predictions = val_predictions

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, train_mse={train_mse:.4f}, train_std={train_std:.4f} | "
            f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, val_mse={val_mse:.4f}, val_std={val_std:.4f}"
        )
        with history_log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(
                f"Epoch {epoch},train_loss={train_loss:.6f},train_mae={train_mae:.6f},train_mse={train_mse:.6f},"
                f"train_std={train_std:.6f},val_loss={val_loss:.6f},val_mae={val_mae:.6f},val_mse={val_mse:.6f},val_std={val_std:.6f}\n"
            )
        history_entries.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": float(np.sqrt(train_mse)),
                "val_rmse": float(np.sqrt(val_mse)),
            }
        )

        # Save the model and evaluation artifacts only if validation loss improves
        if val_loss < best_val_loss:
            improvement = (
                float("inf") if best_val_loss == float("inf") else best_val_loss - val_loss
            )
            best_val_loss = val_loss
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), best_model_path)

            plot_path = output_dir / f"age_val_scatter_epoch{epoch}.png"
            if DisplayUtils.save_regression_scatter(
                val_targets,
                val_predictions,
                save_path=plot_path,
                title=f"Epoch {epoch} Age Predictions (best so far)",
                axis_limits=(0.0, 70.0),
                point_size=20,
                alpha=0.6,
            ):
                print(f"Saved best model to {best_model_path} (val_loss={val_loss:.4f}) and plot to {plot_path}")

            boxplot_path = output_dir / f"age_val_boxplot_epoch{epoch}.png"
            DisplayUtils.save_per_user_boxplot(
                user_ids=val_user_ids,
                targets=val_targets,
                predictions=val_predictions,
                save_path=boxplot_path,
                title=f"Validation per-user box plot (epoch {epoch})",
            )

            val_targets_arr = np.asarray(val_targets, dtype=float)
            val_means_arr = np.asarray(val_predictions, dtype=float)
            val_log_vars_arr = np.asarray(val_log_vars, dtype=float)
            gate_results = compute_age_gate_curves(
                val_targets_arr,
                val_means_arr,
                val_log_vars_arr,
                age_threshold=18.0,
                num_thresholds=201,
            )

            # Derive tau choices and confusion counts (case1: admit adults).
            adult_prob_arr = gate_results["adult_prob"]
            fprs = gate_results["case1"]["fpr"]
            tprs = gate_results["case1"]["tpr"]
            taus = gate_results["case1"]["thresholds"]

            # Closest to top-left corner (0,1)
            dist = np.sqrt((fprs - 0.0) ** 2 + (tprs - 1.0) ** 2)
            idx_best = int(np.argmin(dist))

            # Closest to FPR=0.1
            idx_fpr = int(np.argmin(np.abs(fprs - 0.1)))

            # Closest to TPR=0.9
            idx_tpr = int(np.argmin(np.abs(tprs - 0.9)))

            summary_lines = []
            confusion_points = []
            for label, idx in (
                ("best_topleft", idx_best),
                ("fpr_0.1", idx_fpr),
                ("tpr_0.9", idx_tpr),
            ):
                tau = float(taus[idx])
                tp, tn, fp, fn = _confusion_counts(adult_prob_arr, val_targets_arr, tau, age_threshold=18.0)
                summary_lines.append(f"{label}: tau={tau:.4f}, TP={tp}, TN={tn}, FP={fp}, FN={fn}, FPR={fprs[idx]:.4f}, TPR={tprs[idx]:.4f}")
                confusion_points.append(((fprs[idx], tprs[idx]), f"τ={tau:.3f}"))
            summary_path = output_dir / "confusion_summary.txt"
            with summary_path.open("w", encoding="utf-8") as fp:
                fp.write(f"epoch={epoch}\n")
                fp.write("\n".join(summary_lines))
            with history_log_path.open("a", encoding="utf-8") as log_fp:
                log_fp.write("# Confusion summaries (case1):\n")
                for line in summary_lines:
                    log_fp.write(f"# {line}\n")

            preds_dump_path = output_dir / "best_val_predictions.npz"
            np.savez(
                preds_dump_path,
                targets=val_targets_arr,
                pred_mean=val_means_arr,
                pred_log_var=val_log_vars_arr,
                adult_prob=gate_results["adult_prob"],
                epoch=epoch,
            )

            roc_case1_path = output_dir / f"roc_case1_adult_gate_epoch{epoch}.png"
            roc_case2_path = output_dir / f"roc_case2_child_gate_epoch{epoch}.png"
            DisplayUtils.plot_roc_curve(
                gate_results["case1"]["fpr"],
                gate_results["case1"]["tpr"],
                thresholds=gate_results["case1"]["thresholds"],
                save_path=roc_case1_path,
                title="ROC - Adult Content Gate (admit adults)",
                auc_value=gate_results["case1"]["auc"],
                show=False,
                highlight_points=[pt for pt, _ in confusion_points],
                highlight_labels=[lbl for _, lbl in confusion_points],
            )
            DisplayUtils.plot_roc_curve(
                gate_results["case2"]["fpr"],
                gate_results["case2"]["tpr"],
                thresholds=gate_results["case2"]["thresholds"],
                save_path=roc_case2_path,
                title="ROC - Child Platform Gate (admit minors)",
                auc_value=gate_results["case2"]["auc"],
                show=False,
            )

            metrics_csv_path = output_dir / "age_gate_metrics.csv"
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
                f"Updated ROC plots ({roc_case1_path.name}, {roc_case2_path.name}), "
                f"metrics CSV ({metrics_csv_path.name}), and saved predictions to {preds_dump_path.name}."
            )

            if improvement == float("inf") or improvement >= min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            early_msg = (
                f"Early stopping at epoch {epoch}: validation loss did not improve by at least "
                f"{min_delta:.3f} for {patience} consecutive epochs."
            )
            print(early_msg)
            with history_log_path.open("a", encoding="utf-8") as log_fp:
                log_fp.write(f"# {early_msg}\n")
            history_plot_path = output_dir / "history_plot.png"
            DisplayUtils.plot_loss_history(
                history_entries,
                save_path=history_plot_path,
                show=False,
                title="Training History (MAE & RMSE)",
            )
            print(f"Saved training history plot to {history_plot_path}")
            break

    # Save a final per-user box plot using the last validation epoch data.
    if last_val_user_ids and last_val_targets and last_val_predictions:
        final_boxplot_path = output_dir / "age_val_boxplot_final.png"
        DisplayUtils.save_per_user_boxplot(
            user_ids=last_val_user_ids,
            targets=last_val_targets,
            predictions=last_val_predictions,
            save_path=final_boxplot_path,
            title="Validation per-user box plot (final epoch)",
        )

    print("Training complete. Best model saved on validation improvement.")


if __name__ == "__main__":
    main()
