import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from hands_dataset import get_dataset_root, load_combined_metadata, set_dataset_root
from displayUtils import DisplayUtils
from metrics import LossWeights, weighted_regression_loss
from models import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor

# Example (Windows): python train_age.py --data-root "C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets" --output-dir runs\b4_efficientnet --model b4 --img-size 380 --batch-size 32 --epochs 40 --seed 42 --lr 0.0003

# --- Config -----------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_LR = 3e-4
DEFAULT_MODEL_VARIANT = "b7"
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df[df["age"].notna()]
    df = df.copy()
    df["age"] = df["age"].astype(float)
    return df.reset_index(drop=True)


def stratified_user_split(
    metadata: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    adult_threshold: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique users while preserving the adult/minor ratio when possible."""
    if "user_id" not in metadata.columns or "age" not in metadata.columns:
        raise ValueError("metadata must include 'user_id' and 'age' columns for stratification.")

    per_user = (
        metadata.groupby("user_id")["age"]
        .mean()
        .rename("mean_age")
        .reset_index()
    )
    if per_user.empty:
        raise ValueError("No user records available after filtering; cannot split dataset.")

    labels = (per_user["mean_age"].to_numpy() >= adult_threshold).astype(int)
    user_ids = per_user["user_id"].to_numpy()

    stratify = None
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if unique_labels.size > 1:
        n_test = np.ceil(label_counts * test_size).astype(int)
        n_train = label_counts - n_test
        if np.all(n_test >= 1) and np.all(n_train >= 1):
            stratify = labels

    train_ids, test_ids = train_test_split(
        user_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_ids, test_ids


class AgeDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform=None):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        image_path: Path = row["image_path"]
        age = float(row["age"])
        image = Image.open(image_path).convert("RGB")

        # Optional: crop to square bbox with padding if available
        bbox = row.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox]
                if xmax > xmin and ymax > ymin:
                    w, h = image.size
                    sq_xmin, sq_ymin, sq_xmax, sq_ymax = DisplayUtils.make_square_bbox(
                        (xmin, ymin, xmax, ymax)
                    )

                    # compute required padding to keep crop inside image bounds
                    pad_left = max(0, -sq_xmin)
                    pad_top = max(0, -sq_ymin)
                    pad_right = max(0, sq_xmax - w)
                    pad_bottom = max(0, sq_ymax - h)

                    if pad_left or pad_top or pad_right or pad_bottom:
                        image = ImageOps.expand(
                            image,
                            border=(pad_left, pad_top, pad_right, pad_bottom),
                            fill=(0, 0, 0),
                        )
                        # shift square bbox into padded image coords
                        sq_xmin += pad_left
                        sq_xmax += pad_left
                        sq_ymin += pad_top
                        sq_ymax += pad_top

                    # final safety clamp then crop
                    sq_xmin = max(0, sq_xmin)
                    sq_ymin = max(0, sq_ymin)
                    sq_xmax = max(sq_xmin + 1, min(image.size[0], sq_xmax))
                    sq_ymax = max(sq_ymin + 1, min(image.size[1], sq_ymax))
                    image = image.crop((sq_xmin, sq_ymin, sq_xmax, sq_ymax))
            except Exception:
                # If anything goes wrong with bbox handling, fall back to full image
                pass
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)


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


def build_transforms(img_size: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet hand age regressor.")
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
        "--model",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        choices=sorted(EFFICIENTNET_IMG_SIZES.keys()),
        help="EfficientNet variant to use (default: b7).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override the input resolution (discouraged). By default the canonical size for the chosen model is used.",
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
        "--no-stratified-user-split",
        action="store_true",
        help="Disable per-user stratification when splitting the dataset.",
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
    args = parser.parse_args()

    loss_weights = LossWeights(
        nll=args.loss_weight_nll,
        mse=args.loss_weight_mse,
        mae=args.loss_weight_mae,
    )
    loss_weights.validate()

    set_random_seed(args.seed)
    model_variant = args.model.lower()
    default_size = EFFICIENTNET_IMG_SIZES[model_variant]
    if args.img_size is not None and args.img_size != default_size:
        print(
            f"[train] Ignoring requested --img-size {args.img_size}; "
            f"EfficientNet-{model_variant.upper()} uses {default_size}."
        )
    img_size = default_size

    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = build_transforms(img_size)

    metadata = filter_metadata(load_combined_metadata(root=active_root))
    if args.no_stratified_user_split:
        user_ids = metadata["user_id"].unique()
        train_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=args.seed)
    else:
        train_ids, test_ids = stratified_user_split(
            metadata,
            test_size=0.2,
            random_state=args.seed,
        )
    train_meta = metadata[metadata["user_id"].isin(train_ids)]
    test_meta = metadata[metadata["user_id"].isin(test_ids)]

    print(
        f"Using dataset root: {active_root}\n"
        f"Saving artifacts to: {output_dir}\n"
        f"Train users: {train_meta['user_id'].nunique()} | Train images: {len(train_meta)}\n"
        f"Test users:  {test_meta['user_id'].nunique()} | Test images:  {len(test_meta)}\n"
        f"Model: EfficientNet-{model_variant.upper()} | Image size: {img_size} | Batch size: {args.batch_size}\n"
        f"Epochs: {args.epochs} | Learning rate: {args.lr:.2e} | Seed: {args.seed}"
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

    train_ds = AgeDataset(train_meta, transform=train_transform)
    test_ds = AgeDataset(test_meta, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EfficientNetAgeRegressor(model_variant)
    if DEVICE.type == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Using {gpu_count} GPUs via DataParallel.")
            model = nn.DataParallel(model)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_model_path = output_dir / f"efficientnet_{model_variant}_age_regressor.pth"
    history_log_path = output_dir / "history.log"
    min_delta = 0.001
    patience = 7
    epochs_without_improvement = 0
    history_entries: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        running_std = 0.0
        for images, ages in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, ages = images.to(DEVICE), ages.to(DEVICE)
            optimizer.zero_grad()
            pred_mean, pred_log_var = model(images)
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
        with torch.no_grad():
            for images, ages in test_loader:
                images, ages = images.to(DEVICE), ages.to(DEVICE)
                pred_mean, pred_log_var = model(images)
                batch_loss = weighted_regression_loss(pred_mean, pred_log_var, ages, loss_weights).item()
                val_loss += batch_loss
                val_mae += torch.mean(torch.abs(pred_mean - ages)).item()
                val_mse += torch.mean((pred_mean - ages) ** 2).item()
                val_std += torch.mean(torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))).item()
                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(pred_mean.detach().cpu().tolist())
                val_log_vars.extend(pred_log_var.detach().cpu().tolist())

        denom = max(1, len(test_loader))
        val_loss /= denom
        val_mae /= denom
        val_mse /= denom
        val_std /= denom

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

            preds_dump_path = output_dir / "best_val_predictions.npz"
            np.savez(
                preds_dump_path,
                targets=val_targets_arr,
                pred_mean=val_means_arr,
                pred_log_var=val_log_vars_arr,
                adult_prob=gate_results["adult_prob"],
                epoch=epoch,
            )

            roc_case1_path = output_dir / "roc_case1_adult_gate.png"
            roc_case2_path = output_dir / "roc_case2_child_gate.png"
            DisplayUtils.plot_roc_curve(
                gate_results["case1"]["fpr"],
                gate_results["case1"]["tpr"],
                thresholds=gate_results["case1"]["thresholds"],
                save_path=roc_case1_path,
                title="ROC - Adult Content Gate (admit adults)",
                auc_value=gate_results["case1"]["auc"],
                show=False,
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

    print("Training complete. Best model saved on validation improvement.")


if __name__ == "__main__":
    main()
