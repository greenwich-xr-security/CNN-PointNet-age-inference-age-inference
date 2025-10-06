import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from hands_dataset import load_combined_metadata
from displayUtils import DisplayUtils

# --- Config -----------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 40
LR = 3e-4
IMG_SIZE = 224
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
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)


class ResNetAgeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_feats, 1)

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


metadata = filter_metadata(load_combined_metadata())
user_ids = metadata["user_id"].unique()
train_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=SEED)
train_meta = metadata[metadata["user_id"].isin(train_ids)]
test_meta = metadata[metadata["user_id"].isin(test_ids)]

print(
    f"Train users: {train_meta['user_id'].nunique()} | Train images: {len(train_meta)}\n"
    f"Test users:  {test_meta['user_id'].nunique()} | Test images:  {len(test_meta)}"
)

train_ds = AgeDataset(train_meta, transform=train_transform)
test_ds = AgeDataset(test_meta, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


model = ResNetAgeRegressor().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


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

    train_loss = running_loss / max(1, len(train_loader))

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_targets = []
    val_predictions = []
    with torch.no_grad():
        for images, ages in test_loader:
            images, ages = images.to(DEVICE), ages.to(DEVICE)
            preds = model(images)
            loss = criterion(preds, ages)
            val_loss += loss.item()
            val_mae += torch.mean(torch.abs(preds - ages)).item()
            val_targets.extend(ages.detach().cpu().tolist())
            val_predictions.extend(preds.detach().cpu().tolist())

    val_loss /= max(1, len(test_loader))
    val_mae /= max(1, len(test_loader))

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.2f}")
    DisplayUtils.display_regression_scatter(
        val_targets,
        val_predictions,
        title=f"Epoch {epoch} Age Predictions",
    )


torch.save(model.state_dict(), "resnet18_age_regressor.pth")
print("Model saved to resnet18_age_regressor.pth")
