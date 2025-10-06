"""Utility script for loading and normalising the 11kHands and archive hand datasets.

The module exposes helper functions to load both metadata sources into a unified
Pandas DataFrame as well as a ``HandsDataset`` class that can be re-used by
training or inference scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Base directory for all datasets
ROOT = Path(r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets")

# Subdirectories and CSV files relative to ROOT
PRIMARY_ROOT = ROOT / "11kHands" / "Hands"
PRIMARY_CSV = ROOT / "11kHands" / "HandInfo.csv"
ARCHIVE_ROOT = ROOT / "archive" / "Photos"
ARCHIVE_CSV = ROOT / "archive" / "annotated_dataset_details.csv"

# ---------------------------------------------------------------------------
# Label normalisation helpers
VALID_LABELS = {
    "dorsal left",
    "dorsal right",
    "palmar left",
    "palmar right",
}

LABEL_ALIASES: Dict[str, str] = {
    "dorsal_left": "dorsal left",
    "dorsal-right": "dorsal right",
    "palmar_left": "palmar left",
    "palmar-right": "palmar right",
    "palmer left": "palmar left",
    "palmer right": "palmar right",
}


def _normalise_label(raw: object) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    label = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    label = " ".join(label.split())
    label = LABEL_ALIASES.get(label, label)
    if label in VALID_LABELS:
        return label
    return None


def _normalise_gender(raw: object) -> Optional[str]:
    mapping = {
        "f": "female",
        "female": "female",
        "m": "male",
        "male": "male",
        0: "female",
        1: "female",  # some datasets use 1 for female
        2: "male",
    }
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        raw_lower = str(raw).strip().lower()
    except Exception:  # noqa: BLE001
        return None
    return mapping.get(raw_lower, mapping.get(raw, None))


# ---------------------------------------------------------------------------
# Metadata loaders

def _build_archive_filename(person_no: int, age: int, gender: Optional[int], photo_no: int) -> Optional[Path]:
    parts = [str(person_no), str(age)]
    if gender is not None:
        parts.append(str(gender))
    parts.append(str(photo_no))
    stem = "_".join(parts)
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = ARCHIVE_ROOT / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_primary_metadata() -> pd.DataFrame:
    if not PRIMARY_CSV.exists():
        raise FileNotFoundError(f"Primary CSV not found: {PRIMARY_CSV}")

    df = pd.read_csv(PRIMARY_CSV)
    df["aspect_norm"] = df["aspectOfHand"].apply(_normalise_label)
    df = df[df["aspect_norm"].notna()]

    df["image_path"] = df["imageName"].apply(lambda name: PRIMARY_ROOT / str(name))
    df = df[df["image_path"].apply(Path.exists)]

    df["gender_norm"] = df["gender"].apply(_normalise_gender)
    df["age_norm"] = df["age"].apply(lambda x: int(x) if pd.notna(x) else pd.NA)

    df_out = pd.DataFrame(
        {
            "source": "primary",
            "user_id": df["id"].apply(lambda x: f"primary_{int(x)}"),
            "age": df["age_norm"],
            "gender": df["gender_norm"],
            "aspect": df["aspect_norm"],
            "image_path": df["image_path"].apply(Path),
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"Primary dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_archive_metadata() -> pd.DataFrame:
    if not ARCHIVE_CSV.exists():
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path"])

    df = pd.read_csv(ARCHIVE_CSV)
    if df.empty or "aspectOfHand" not in df.columns:
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path"])

    df["aspect_norm"] = df["aspectOfHand"].apply(_normalise_label)
    df = df[df["aspect_norm"].notna()]

    def resolve_path(row) -> Optional[Path]:
        try:
            person_no = int(row.get("Person No"))
            age_val = row.get("Age", -1)
            age = int(age_val) if pd.notna(age_val) else -1
            gender_val = row.get("Gender", None)
            gender = int(gender_val) if pd.notna(gender_val) else None
            photo_no = int(row.get("Photo No"))
        except (TypeError, ValueError):
            return None
        return _build_archive_filename(person_no, age, gender, photo_no)

    df["image_path"] = df.apply(resolve_path, axis=1)
    df = df[df["image_path"].notna()]

    gender_map = {1: "female", 2: "male"}
    df["gender_norm"] = df["Gender"].apply(lambda g: gender_map.get(g) if pd.notna(g) else None)
    df["age_norm"] = df["Age"].apply(lambda a: int(a) if pd.notna(a) else pd.NA)

    df_out = pd.DataFrame(
        {
            "source": "archive",
            "user_id": df["Person No"].apply(lambda x: f"archive_{int(x)}"),
            "age": df["age_norm"],
            "gender": df["gender_norm"],
            "aspect": df["aspect_norm"],
            "image_path": df["image_path"].apply(Path),
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"Archive dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_combined_metadata() -> pd.DataFrame:
    primary_df = load_primary_metadata()
    archive_df = load_archive_metadata()
    combined = pd.concat([primary_df, archive_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="image_path")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset class

class HandsDataset(Dataset):
    """Simple dataset that yields an image tensor, its label, and metadata."""

    def __init__(self, records: pd.DataFrame, transform=None):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401
        return len(self.records)

    def __getitem__(self, idx: int):  # noqa: D401
        row = self.records.iloc[idx]
        image_path: Path = row["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load image {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        label: str = row["aspect"]
        metadata = {
            "user_id": row["user_id"],
            "age": row["age"],
            "gender": row["gender"],
            "source": row["source"],
            "image_path": image_path,
        }
        return image, label, metadata


if __name__ == "__main__":
    combined = load_combined_metadata()
    print(f"Combined samples: {len(combined)}")
    print(combined.groupby(["source", "aspect"]).size())