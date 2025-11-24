"""Utility script for loading and normalising the 11kHands and archive hand datasets.

The module exposes helper functions to load both metadata sources into a unified
Pandas DataFrame as well as a ``HandsDataset`` class that can be re-used by
training or inference scripts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Base directory (can be overridden via env var or function argument)
_DEFAULT_ROOT = Path(r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets")
_ENV_VAR_NAME = "HANDS_DATASETS_ROOT"

PathLike = Union[str, Path]

_DATA_ROOT = Path(os.environ.get(_ENV_VAR_NAME, _DEFAULT_ROOT))


def set_dataset_root(root: PathLike) -> Path:
    """Override the dataset root used by loader helpers."""
    global _DATA_ROOT
    _DATA_ROOT = Path(root).expanduser()
    return _DATA_ROOT


def get_dataset_root() -> Path:
    """Return the currently configured dataset root."""
    return _DATA_ROOT


def _resolve_root(root: Optional[PathLike] = None) -> Path:
    if root is None:
        return get_dataset_root()
    return Path(root).expanduser()

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
    "right dorsal": "dorsal right",
    "left dorsal": "dorsal left",
    "right palmar": "palmar right",
    "left palmar": "palmar left",
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

def _build_archive_filename(
    person_no: int,
    age: int,
    gender: Optional[int],
    photo_no: int,
    *,
    archive_root: Path,
) -> Optional[Path]:
    parts = [str(person_no), str(age)]
    if gender is not None:
        parts.append(str(gender))
    parts.append(str(photo_no))
    stem = "_".join(parts)
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = archive_root / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_primary_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    dataset_root = _resolve_root(root)
    primary_root = dataset_root / "11kHands" / "Hands"
    primary_csv = dataset_root / "11kHands" / "HandInfo.csv"

    if not primary_csv.exists():
        raise FileNotFoundError(f"Primary CSV not found: {primary_csv}")

    raw_df = pd.read_csv(primary_csv)
    working_df = raw_df.copy()

    working_df["aspect_norm"] = working_df["aspectOfHand"].apply(_normalise_label)
    working_df = working_df[working_df["aspect_norm"].notna()]

    working_df["image_path"] = working_df["imageName"].apply(lambda name: primary_root / str(name))
    working_df = working_df[working_df["image_path"].apply(Path.exists)]

    working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    working_df["age_norm"] = working_df["age"].apply(lambda x: int(x) if pd.notna(x) else pd.NA)

    df_out = pd.DataFrame(
        {
            "source": "primary",
            "user_id": working_df["id"].apply(lambda x: f"primary_{int(x)}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"Primary dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_archive_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    dataset_root = _resolve_root(root)
    archive_root = dataset_root / "archive" / "Photos"
    archive_csv = dataset_root / "archive" / "annotated_dataset_details.csv"

    if not archive_csv.exists():
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path"])

    raw_df = pd.read_csv(archive_csv)
    if raw_df.empty or "aspectOfHand" not in raw_df.columns:
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path"])

    working_df = raw_df.copy()
    working_df["aspect_norm"] = working_df["aspectOfHand"].apply(_normalise_label)
    working_df = working_df[working_df["aspect_norm"].notna()]

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
        return _build_archive_filename(
            person_no,
            age,
            gender,
            photo_no,
            archive_root=archive_root,
        )

    working_df["image_path"] = working_df.apply(resolve_path, axis=1)
    working_df = working_df[working_df["image_path"].notna()]

    gender_map = {1: "female", 2: "male"}
    working_df["gender_norm"] = working_df["Gender"].apply(lambda g: gender_map.get(g) if pd.notna(g) else None)
    working_df["age_norm"] = working_df["Age"].apply(lambda a: int(a) if pd.notna(a) else pd.NA)

    df_out = pd.DataFrame(
        {
            "source": "archive",
            "user_id": working_df["Person No"].apply(lambda x: f"archive_{int(x)}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"].apply(Path),
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"Archive dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_combined_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    primary_df = load_primary_metadata(root=root)
    archive_df = load_archive_metadata(root=root)
    combined = pd.concat([primary_df, archive_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="image_path")
    return combined.reset_index(drop=True)


def load_handrgbd_metadata(
    root: Optional[PathLike] = None,
    *,
    aspect_filter: Optional[Union[str, list[str], tuple[str, ...]]] = None,
) -> pd.DataFrame:
    """Load metadata for the handRGBD dataset (RGB + XYZ EXR pairs).

    Returns rows with existing RGB/EXR files and normalised labels.
    aspect_filter can be a single string or list/tuple; values matching a full
    aspect label (e.g., "dorsal right") or a keyword ("dorsal") are kept.
    """
    dataset_root = _resolve_root(root)
    hand_root = dataset_root / "handRGBD"
    csv_path = hand_root / "reference_table.csv"
    rgb_dir = hand_root / "rgb"
    xyz_dir = hand_root / "xyz"

    if not csv_path.exists():
        raise FileNotFoundError(f"handRGBD reference CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    required_cols = ["user_id", "age", "gender", "aspect", "name", "lights", "wall"]
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"handRGBD CSV missing required columns: {missing_cols}")

    working_df = raw_df[required_cols].copy()
    working_df["aspect_norm"] = working_df["aspect"].apply(_normalise_label)
    working_df = working_df[working_df["aspect_norm"].notna()]

    working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    working_df["age_norm"] = working_df["age"].apply(lambda x: int(x) if pd.notna(x) else pd.NA)

    working_df["rgb_path"] = working_df["name"].apply(lambda n: rgb_dir / f"{n}.png")
    working_df["xyz_path"] = working_df["name"].apply(lambda n: xyz_dir / f"{n}.exr")

    file_mask = working_df["rgb_path"].apply(Path.exists) & working_df["xyz_path"].apply(Path.exists)
    missing_count = int((~file_mask).sum())
    if missing_count:
        print(f"[handRGBD] Skipping {missing_count} entries with missing RGB or EXR files.")
    working_df = working_df[file_mask]

    if aspect_filter is not None:
        if isinstance(aspect_filter, str):
            requested = [aspect_filter]
        else:
            requested = list(aspect_filter)
        canonical: set[str] = set()
        keywords: set[str] = set()
        for raw in requested:
            if raw is None:
                continue
            norm = _normalise_label(raw)
            if norm:
                canonical.add(norm)
                continue
            raw_lower = str(raw).strip().lower()
            if raw_lower:
                keywords.add(raw_lower)

        def _aspect_matches(val: str) -> bool:
            if val in canonical:
                return True
            return any(key in val for key in keywords)

        before = len(working_df)
        working_df = working_df[working_df["aspect_norm"].apply(_aspect_matches)]
        print(f"[handRGBD] Aspect filter kept {len(working_df)} / {before} rows.")

    df_out = pd.DataFrame(
        {
            "source": "handRGBD",
            "user_id": working_df["user_id"].apply(lambda u: f"handrgbd_{int(u)}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "lights": working_df["lights"],
            "wall": working_df["wall"],
            "name": working_df["name"],
            "rgb_path": working_df["rgb_path"],
            "xyz_path": working_df["xyz_path"],
            # keep compatibility with existing datasets
            "image_path": working_df["rgb_path"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"handRGBD dataset -> users: {df_out['user_id'].nunique()} | pairs: {len(df_out)}"
    )
    return df_out


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
    import argparse

    parser = argparse.ArgumentParser(description="Inspect combined hands dataset metadata.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=f"Path to the dataset root (overrides env var {_ENV_VAR_NAME}).",
    )
    args = parser.parse_args()

    combined = load_combined_metadata(root=args.root)
    active_root = _resolve_root(args.root)
    print(f"Using dataset root: {active_root}")
    print(f"Combined samples: {len(combined)}")
    print(combined.groupby(["source", "aspect"]).size())
