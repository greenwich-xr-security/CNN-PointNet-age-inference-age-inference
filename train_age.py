import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from hands_dataset import get_dataset_root, load_combined_metadata, set_dataset_root
from displayUtils import DisplayUtils

# Example (Windows): python train_age.py --data-root "C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets" --output-dir runs\b4_efficientnet --model b4 --img-size 380 --batch-size 32 --epochs 40 --seed 42 --lr 0.0003

# --- Config -----------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_LR = 3e-4
DEFAULT_MODEL_VARIANT = "b7"
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EFFICIENTNET_IMG_SIZES = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
    "b3": 300,
    "b4": 380,
    "b5": 456,
    "b6": 528,
    "b7": 600,
}

def get_default_efficientnet_weights(variant: str):
    """Resolve the torchvision weights enum for the requested EfficientNet variant."""
    weights_enum_name = f"EfficientNet_{variant.upper()}_Weights"
    weights_enum = getattr(models, weights_enum_name, None)
    if weights_enum is None:
        return None
    default_weights = getattr(weights_enum, "DEFAULT", None)
    if default_weights is not None:
        return default_weights
    try:
        return next(iter(weights_enum))
    except TypeError:
        return None


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
                    # compute square around bbox center
                    bw = xmax - xmin
                    bh = ymax - ymin
                    side = int(max(bw, bh))
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    sq_xmin = int(np.floor(cx - side / 2.0))
                    sq_ymin = int(np.floor(cy - side / 2.0))
                    sq_xmax = sq_xmin + side
                    sq_ymax = sq_ymin + side

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


class EfficientNetAgeRegressor(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        variant = variant.lower()
        if variant not in EFFICIENTNET_IMG_SIZES:
            raise ValueError(f"Unsupported EfficientNet variant '{variant}'.")

        model_name = f"efficientnet_{variant}"
        if not hasattr(models, model_name):
            raise ValueError(f"torchvision.models does not provide '{model_name}'.")

        backbone_builder = getattr(models, model_name)

        weights = get_default_efficientnet_weights(variant)
        try:
            if weights is not None:
                backbone = backbone_builder(weights=weights)
            else:
                backbone = backbone_builder(pretrained=True)
        except TypeError:
            backbone = backbone_builder(pretrained=True)

        self.backbone = backbone
        self.variant = variant
        # Replace the classifier to output a single regression value
        if isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) >= 2:
            in_feats = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_feats, 1)
        else:
            # Fallback: handle unexpected classifier structure
            in_feats = getattr(self.backbone.classifier, 'in_features', None)
            if in_feats is None:
                raise RuntimeError(f"Unexpected EfficientNet-{variant.upper()} classifier structure")
            self.backbone.classifier = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


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
        help="Override the input resolution. Defaults to the canonical size for the chosen model.",
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
    args = parser.parse_args()

    set_random_seed(args.seed)
    model_variant = args.model.lower()
    default_size = EFFICIENTNET_IMG_SIZES[model_variant]
    img_size = args.img_size or default_size

    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = build_transforms(img_size)

    metadata = filter_metadata(load_combined_metadata(root=active_root))
    user_ids = metadata["user_id"].unique()
    train_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=args.seed)
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
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_model_path = output_dir / f"efficientnet_{model_variant}_age_regressor.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        for images, ages in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, ages = images.to(DEVICE), ages.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            mae = torch.mean(torch.abs(preds - ages)).item()
            mse = torch.mean((preds - ages) ** 2).item()
            running_mae += mae
            running_mse += mse

        denom = max(1, len(train_loader))
        train_loss = running_loss / denom
        train_mae = running_mae / denom
        train_mse = running_mse / denom

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for images, ages in test_loader:
                images, ages = images.to(DEVICE), ages.to(DEVICE)
                preds = model(images)
                batch_loss = criterion(preds, ages).item()
                val_loss += batch_loss
                val_mae += torch.mean(torch.abs(preds - ages)).item()
                val_mse += torch.mean((preds - ages) ** 2).item()
                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(preds.detach().cpu().tolist())

        denom = max(1, len(test_loader))
        val_loss /= denom
        val_mae /= denom
        val_mse /= denom

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, train_mse={train_mse:.4f} | "
            f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, val_mse={val_mse:.4f}"
        )

        # Save the model and a scatter plot only if validation loss improves
        if val_loss < best_val_loss:
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

    print("Training complete. Best model saved on validation improvement.")


if __name__ == "__main__":
    main()
