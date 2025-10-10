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
from matplotlib import pyplot as plt

from hands_dataset import get_dataset_root, load_combined_metadata, set_dataset_root
from displayUtils import DisplayUtils

# --- Config -----------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 40
LR = 3e-4
IMG_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


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
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b3(pretrained=True)
        # Replace the classifier to output a single regression value
        if isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) >= 2:
            in_feats = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_feats, 1)
        else:
            # Fallback: handle unexpected classifier structure
            in_feats = getattr(self.backbone.classifier, 'in_features', None)
            if in_feats is None:
                raise RuntimeError('Unexpected EfficientNet-B2 classifier structure')
            self.backbone.classifier = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
    args = parser.parse_args()

    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = filter_metadata(load_combined_metadata(root=active_root))
    user_ids = metadata["user_id"].unique()
    train_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=SEED)
    train_meta = metadata[metadata["user_id"].isin(train_ids)]
    test_meta = metadata[metadata["user_id"].isin(test_ids)]

    print(
        f"Using dataset root: {active_root}\n"
        f"Saving artifacts to: {output_dir}\n"
        f"Train users: {train_meta['user_id'].nunique()} | Train images: {len(train_meta)}\n"
        f"Test users:  {test_meta['user_id'].nunique()} | Test images:  {len(test_meta)}"
    )

    train_ds = AgeDataset(train_meta, transform=train_transform)
    test_ds = AgeDataset(test_meta, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EfficientNetAgeRegressor().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_mae = float("inf")
    best_model_path = output_dir / "efficientnet_b2_age_regressor.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, ages in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, ages = images.to(DEVICE), ages.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_mae = running_loss / max(1, len(train_loader))

        model.eval()
        val_mae = 0.0
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for images, ages in test_loader:
                images, ages = images.to(DEVICE), ages.to(DEVICE)
                preds = model(images)
                val_mae += torch.mean(torch.abs(preds - ages)).item()
                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(preds.detach().cpu().tolist())

        val_mae /= max(1, len(test_loader))

        print(f"Epoch {epoch}: train_mae={train_mae:.4f}, val_mae={val_mae:.4f}")

        # Save the model and a scatter plot only if validation loss improves
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)

            # Save scatter plot without displaying
            targets_arr = np.asarray(val_targets, dtype=float)
            preds_arr = np.asarray(val_predictions, dtype=float)
            if targets_arr.size > 0 and preds_arr.size > 0:
                axis_min, axis_max = 0.0, 70.0

                plt.figure(figsize=(6, 6))
                plt.scatter(targets_arr, preds_arr, s=20, alpha=0.6, edgecolors="none")
                plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1)
                plt.xlabel("True Age")
                plt.ylabel("Predicted Age")
                plt.title(f"Epoch {epoch} Age Predictions (best so far)")
                plt.xlim(axis_min, axis_max)
                plt.ylim(axis_min, axis_max)
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
                plot_path = output_dir / f"age_val_scatter_epoch{epoch}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved best model to {best_model_path} (val_mae={val_mae:.4f}) and plot to {plot_path}")

    print("Training complete. Best model saved on validation improvement.")


if __name__ == "__main__":
    main()
