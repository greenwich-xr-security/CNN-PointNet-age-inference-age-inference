import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# === CONFIG ===
PRIMARY_ROOT = Path(r"C:\Users\Staff\Documents\HandsDatasets\11kHands\Hands")
PRIMARY_CSV = Path(r"C:\Users\Staff\Documents\HandsDatasets\11kHands\HandInfo.csv")
ARCHIVE_ROOT = Path(r"C:\Users\Staff\Documents\HandsDatasets\archive\Photos")
ARCHIVE_CSV = Path(r"C:\Users\Staff\Documents\HandsDatasets\archive\annotated_dataset_details.csv")
BATCH_SIZE = 32
BASE_EPOCHS = 10
MAX_EPOCHS = 100
LR = 1e-4
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_ACCURACY = 0.98
ARCHIVE_TARGET_ACCURACY = 0.98

VALID_LABELS = [
    "dorsal left",
    "dorsal right",
    "palmar left",
    "palmar right",
]


# === DATA HELPERS ===
def build_archive_filename(person_no, age, gender, photo_no):
    """Reconstruct archive filename from metadata fields."""
    parts = [str(person_no), str(age)]
    if gender is not None and not pd.isna(gender):
        parts.append(str(int(gender)))
    parts.append(str(photo_no))
    base = "_".join(parts)
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = ARCHIVE_ROOT / f"{base}{ext}"
        if candidate.is_file():
            return candidate
    return None

def load_primary_dataset():
    if not PRIMARY_CSV.exists():
        raise FileNotFoundError(f"Primary CSV not found: {PRIMARY_CSV}")
    df = pd.read_csv(PRIMARY_CSV)
    df = df[df["aspectOfHand"].str.lower().isin(VALID_LABELS)]
    df["image_path"] = df["imageName"].apply(lambda name: PRIMARY_ROOT / name)
    df = df[df["image_path"].apply(lambda p: p.is_file())]
    df["label"] = df["aspectOfHand"].str.lower()
    df["split_id"] = df["id"].astype(str)
    return df[["image_path", "label", "split_id"]]

def load_archive_dataset():
    if not ARCHIVE_CSV.exists():
        return pd.DataFrame(columns=["image_path", "label", "split_id"])
    try:
        df = pd.read_csv(ARCHIVE_CSV)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read {ARCHIVE_CSV}: {exc}")
        return pd.DataFrame(columns=["image_path", "label", "split_id"])
    if df.empty or "aspectOfHand" not in df.columns:
        return pd.DataFrame(columns=["image_path", "label", "split_id"])
    df = df[df["aspectOfHand"].str.lower().isin(VALID_LABELS)]

    def resolve_path(row):
        person_no = row.get("Person No")
        photo_no = row.get("Photo No")
        if pd.isna(person_no) or pd.isna(photo_no):
            return None
        age = row.get("Age", -1)
        gender = row.get("Gender", None)
        try:
            person_no = int(person_no)
            age = int(age) if not pd.isna(age) else -1
            gender_val = int(gender) if not pd.isna(gender) else None
            photo_no = int(photo_no)
        except (ValueError, TypeError):
            return None
        return build_archive_filename(person_no, age, gender_val, photo_no)

    df["image_path"] = df.apply(resolve_path, axis=1)
    df = df[df["image_path"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["image_path", "label", "split_id"])
    df = df.drop_duplicates(subset="image_path")
    df["label"] = df["aspectOfHand"].str.lower()
    df["split_id"] = "archive_" + df["Person No"].astype(int).astype(str)
    return df[["image_path", "label", "split_id"]]

def split_datasets(df):
    unique_ids = df["split_id"].unique()
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=0.2,
        random_state=42,
    )
    train_df = df[df["split_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["split_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


# === DATASET ===
class HandsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _parse_labels(label_str):
        label_str = label_str.strip().lower()
        is_dorsal = 0 if "dorsal" in label_str else 1  # 0: dorsal, 1: palmar
        is_left = 0 if "left" in label_str else 1      # 0: left, 1: right
        return is_dorsal, is_left

    def __getitem__(self, idx):
        while True:
            row = self.df.iloc[idx]
            img_path = Path(row["image_path"])
            label = row["label"]
            dorsal_palmar_label, left_right_label = self._parse_labels(label)

            if not img_path.exists():
                idx = (idx + 1) % len(self.df)
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError):
                idx = (idx + 1) % len(self.df)
                continue

            if self.transform:
                img = self.transform(img)

            return (
                img,
                torch.tensor(dorsal_palmar_label, dtype=torch.long),
                torch.tensor(left_right_label, dtype=torch.long),
            )


# === LOAD & MERGE DATA ===
primary_df = load_primary_dataset()
archive_train_df = load_archive_dataset()

combined_df = pd.concat([primary_df, archive_train_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset="image_path")
if combined_df.empty:
    raise ValueError("No training data found after combining datasets.")

train_df, test_df = split_datasets(combined_df)

print(
    f"Training records -> total: {len(combined_df)}, train: {len(train_df)}, test: {len(test_df)}"
)
print(
    f"Unique IDs -> total: {combined_df['split_id'].nunique()}, "
    f"train: {train_df['split_id'].nunique()}, test: {test_df['split_id'].nunique()}"
)

archive_eval_df = archive_train_df[["image_path", "label"]].copy()
if archive_eval_df.empty:
    print("Archive evaluation dataset not available or empty.")
else:
    print(f"Archive evaluation samples: {len(archive_eval_df)}")


# === AUGMENTATIONS ===
class RandomColorInvert:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return TF.invert(img)
        return img


# === TRANSFORMS ===
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.6), ratio=(0.8, 1.5)),
    transforms.RandomRotation(degrees=(-180, 180)),
    RandomColorInvert(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = HandsDataset(train_df, transform=train_transform)
test_dataset = HandsDataset(test_df, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# === MODEL ===
class ResNetMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dorsal_palmar_head = nn.Linear(num_features, 2)
        self.left_right_head = nn.Linear(num_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        dorsal_palmar_logits = self.dorsal_palmar_head(features)
        left_right_logits = self.left_right_head(features)
        return dorsal_palmar_logits, left_right_logits


model = ResNetMultiHead().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def compute_metrics(dp_logits, lr_logits, dp_labels, lr_labels):
    dp_preds = torch.argmax(dp_logits, dim=1)
    lr_preds = torch.argmax(lr_logits, dim=1)
    dp_correct = (dp_preds == dp_labels).sum().item()
    lr_correct = (lr_preds == lr_labels).sum().item()
    both_correct = ((dp_preds == dp_labels) & (lr_preds == lr_labels)).sum().item()
    return dp_correct, lr_correct, both_correct


def evaluate_on_archive(model):
    if archive_eval_df.empty:
        print("Archive evaluation dataset not available or empty; skipping check.")
        return None

    model.eval()
    dp_correct = 0
    lr_correct = 0
    both_correct = 0
    total = 0

    for _, row in archive_eval_df.iterrows():
        img_path = Path(row["image_path"])
        label = row["label"]
        if not img_path.is_file():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            continue

        dp_label = 0 if "dorsal" in label else 1
        lr_label = 0 if "left" in label else 1

        img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dp_logits, lr_logits = model(img_tensor)
            dp_pred = torch.argmax(dp_logits, dim=1).item()
            lr_pred = torch.argmax(lr_logits, dim=1).item()

        dp_correct += int(dp_pred == dp_label)
        lr_correct += int(lr_pred == lr_label)
        both_correct += int(dp_pred == dp_label and lr_pred == lr_label)
        total += 1

    if total == 0:
        print("Archive evaluation skipped: no valid images found.")
        return None

    dp_acc = dp_correct / total
    lr_acc = lr_correct / total
    both_acc = both_correct / total
    print(
        f"Archive Evaluation -> Dorsal/Palmar Acc: {dp_acc:.3f}, "
        f"Left/Right Acc: {lr_acc:.3f}, Both Correct: {both_acc:.3f}"
    )
    return dp_acc, lr_acc, both_acc


# === TRAIN LOOP ===
archive_target_reached = False
best_archive_acc = 0.0
epoch = 0

while True:
    epoch += 1
    model.train()
    running_loss = 0.0
    dp_correct = 0
    lr_correct = 0
    both_correct = 0
    total = 0

    for imgs, dp_labels, lr_labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs = imgs.to(DEVICE)
        dp_labels = dp_labels.to(DEVICE)
        lr_labels = lr_labels.to(DEVICE)

        optimizer.zero_grad()
        dp_logits, lr_logits = model(imgs)
        loss = criterion(dp_logits, dp_labels) + criterion(lr_logits, lr_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_dp_correct, batch_lr_correct, batch_both_correct = compute_metrics(
            dp_logits, lr_logits, dp_labels, lr_labels
        )
        dp_correct += batch_dp_correct
        lr_correct += batch_lr_correct
        both_correct += batch_both_correct
        total += dp_labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_dp_acc = dp_correct / total
    train_lr_acc = lr_correct / total
    train_both_acc = both_correct / total
    print(
        f"Train Loss: {train_loss:.4f} | Dorsal/Palmar Acc: {train_dp_acc:.3f} | "
        f"Left/Right Acc: {train_lr_acc:.3f} | Both Correct Acc: {train_both_acc:.3f}"
    )

    model.eval()
    dp_correct = 0
    lr_correct = 0
    both_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, dp_labels, lr_labels in test_loader:
            imgs = imgs.to(DEVICE)
            dp_labels = dp_labels.to(DEVICE)
            lr_labels = lr_labels.to(DEVICE)

            dp_logits, lr_logits = model(imgs)
            batch_dp_correct, batch_lr_correct, batch_both_correct = compute_metrics(
                dp_logits, lr_logits, dp_labels, lr_labels
            )
            dp_correct += batch_dp_correct
            lr_correct += batch_lr_correct
            both_correct += batch_both_correct
            total += dp_labels.size(0)

    val_dp_acc = dp_correct / total if total > 0 else 0.0
    val_lr_acc = lr_correct / total if total > 0 else 0.0
    val_both_acc = both_correct / total if total > 0 else 0.0

    print(
        f"Validation Dorsal/Palmar Acc: {val_dp_acc:.3f} | "
        f"Validation Left/Right Acc: {val_lr_acc:.3f} | "
        f"Validation Both Correct Acc: {val_both_acc:.3f}"
    )

    archive_metrics = evaluate_on_archive(model)
    if archive_metrics is not None:
        _, _, archive_both_acc = archive_metrics
        best_archive_acc = max(best_archive_acc, archive_both_acc)
        if archive_both_acc >= ARCHIVE_TARGET_ACCURACY:
            print(
                f"Archive target accuracy {ARCHIVE_TARGET_ACCURACY:.2f} achieved after epoch {epoch}."
            )
            archive_target_reached = True
            break

    if epoch >= MAX_EPOCHS:
        print(
            f"Reached MAX_EPOCHS={MAX_EPOCHS} without hitting archive target. "
            f"Best archive both accuracy: {best_archive_acc:.3f}."
        )
        break

    if epoch >= BASE_EPOCHS and archive_metrics is None:
        print(
            "Stopping because archive evaluation is unavailable after base epochs."
        )
        break

model_path = "resnet18_hands_multilabel.pth"
torch.save(model.state_dict(), model_path)

if archive_target_reached:
    print("Archive target met; training halted and model saved.")
else:
    print("Model saved; archive target not met within limits.")
