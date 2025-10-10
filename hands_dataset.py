"""Utility script for loading and normalising the 11kHands and archive hand datasets.

The module exposes helper functions to load both metadata sources into a unified
Pandas DataFrame as well as a ``HandsDataset`` class that can be re-used by
training or inference scripts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from handLandmarks.handLandmarksDetection import (
    MediaPipeTaskHandLandmarkDetector,
    SentisHandLandmarkDetector,
)

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
# Bounding-box helpers

_MP_DETECTOR: Optional[MediaPipeTaskHandLandmarkDetector] = None
_SENTIS_DETECTOR: Optional[SentisHandLandmarkDetector] = None


def _get_mediapipe_detector() -> Optional[MediaPipeTaskHandLandmarkDetector]:
    global _MP_DETECTOR
    if _MP_DETECTOR is not None:
        return _MP_DETECTOR
    try:
        _MP_DETECTOR = MediaPipeTaskHandLandmarkDetector()
    except FileNotFoundError as exc:
        print(f"[bbox] MediaPipe detector unavailable: {exc}")
        _MP_DETECTOR = None
    return _MP_DETECTOR


def _get_sentis_detector() -> Optional[SentisHandLandmarkDetector]:
    global _SENTIS_DETECTOR
    if _SENTIS_DETECTOR is not None:
        return _SENTIS_DETECTOR
    try:
        _SENTIS_DETECTOR = SentisHandLandmarkDetector()
    except Exception as exc:  # noqa: BLE001
        print(f"[bbox] Sentis detector unavailable: {exc}")
        _SENTIS_DETECTOR = None
    return _SENTIS_DETECTOR


def _parse_bbox(raw: object) -> Optional[Tuple[int, int, int, int]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        cleaned = (
            raw.strip()
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
        )
        if not cleaned:
            return None
        cleaned = cleaned.replace(";", ",")
        if "," in cleaned:
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in cleaned.split() if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        return None

    if len(parts) != 4:
        return None
    try:
        values = tuple(int(round(float(p))) for p in parts)
    except (TypeError, ValueError):
        return None
    return values  # xmin, ymin, xmax, ymax


def _bbox_to_string(bbox: Optional[Tuple[int, int, int, int]]) -> str:
    if bbox is None:
        return ""
    return ",".join(str(int(v)) for v in bbox)


def _compute_bbox_for_image(image_path: Path) -> Optional[Tuple[int, int, int, int]]:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[bbox] Warning: failed to read image '{image_path}'")
        return None

    height, width = img.shape[:2]

    def _detect_with(detector) -> Optional[np.ndarray]:
        if detector is None:
            return None
        try:
            landmarks_norm, _ = detector.detect(img)
        except Exception as exc:  # noqa: BLE001
            print(f"[bbox] {detector.__class__.__name__} failed on '{image_path}': {exc}")
            return None
        if landmarks_norm is None:
            return None
        arr = np.asarray(landmarks_norm, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr

    landmarks_norm = _detect_with(_get_mediapipe_detector())
    fallback_used = False

    if landmarks_norm is None:
        fallback_used = True
        landmarks_norm = _detect_with(_get_sentis_detector())

    if landmarks_norm is None:
        print(f"[bbox] Warning: no hand detected in '{image_path}'")
        return None

    xs = np.clip(landmarks_norm[:, 0], 0.0, 1.0) * max(width - 1, 0)
    ys = np.clip(landmarks_norm[:, 1], 0.0, 1.0) * max(height - 1, 0)

    xmin = int(np.floor(xs.min()))
    xmax = int(np.ceil(xs.max()))
    ymin = int(np.floor(ys.min()))
    ymax = int(np.ceil(ys.max()))

    if fallback_used:
        print(f"[bbox] Fallback detector succeeded for '{image_path}'")

    return xmin, ymin, xmax, ymax


def _ensure_bboxes(
    df: pd.DataFrame,
    image_path_col: str,
    *,
    csv_source: Optional[Path] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    if "bbox" in df.columns:
        parsed = df["bbox"].apply(_parse_bbox)
    else:
        parsed = pd.Series(index=df.index, data=[None] * len(df), dtype="object")

    missing_mask = parsed.isna()
    if not missing_mask.any():
        return parsed

    updates_for_csv: Dict[int, str] = {}

    if raw_df is not None and "bbox" not in raw_df.columns:
        raw_df["bbox"] = ""

    for idx, image_path in df.loc[missing_mask, image_path_col].items():
        bbox = _compute_bbox_for_image(Path(image_path))
        parsed.at[idx] = bbox
        if raw_df is not None:
            formatted = _bbox_to_string(bbox)
            current = raw_df.at[idx, "bbox"] if idx in raw_df.index and "bbox" in raw_df.columns else ""
            if formatted != current:
                raw_df.at[idx, "bbox"] = formatted
                updates_for_csv[idx] = formatted

    if updates_for_csv and csv_source is not None:
        raw_df.to_csv(csv_source, index=False)
        source_label = csv_source.name if hasattr(csv_source, "name") else str(csv_source)
        print(f"[bbox] Stored {len(updates_for_csv)} computed bounding boxes in {source_label}")

    return parsed


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

    working_df["bbox_tuple"] = _ensure_bboxes(
        working_df,
        "image_path",
        csv_source=primary_csv,
        raw_df=raw_df,
    )

    df_out = pd.DataFrame(
        {
            "source": "primary",
            "user_id": working_df["id"].apply(lambda x: f"primary_{int(x)}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"],
            "bbox": working_df["bbox_tuple"],
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
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path", "bbox"])

    raw_df = pd.read_csv(archive_csv)
    if raw_df.empty or "aspectOfHand" not in raw_df.columns:
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path", "bbox"])

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

    working_df["bbox_tuple"] = _ensure_bboxes(
        working_df,
        "image_path",
        csv_source=archive_csv,
        raw_df=raw_df,
    )

    df_out = pd.DataFrame(
        {
            "source": "archive",
            "user_id": working_df["Person No"].apply(lambda x: f"archive_{int(x)}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"].apply(Path),
            "bbox": working_df["bbox_tuple"],
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
            "bbox": row.get("bbox"),
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
