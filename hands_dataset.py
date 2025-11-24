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

    # Accept labels regardless of token order (e.g., "right palmar" or "palmar right").
    tokens = label.split()
    side = None
    surface = None
    for tok in tokens:
        if tok in ("left", "right"):
            side = tok
        elif tok in ("dorsal", "palmar", "palmer", "palm"):
            surface = "palmar" if tok in ("palmer", "palm") else tok
    if surface and side:
        candidate = f"{surface} {side}"
        if candidate in VALID_LABELS:
            return candidate

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


def _center_square_bbox(width: int, height: int) -> Tuple[int, int, int, int]:
    """Return a square bbox centered within the given width/height."""
    side = min(width, height)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return x1, y1, x2, y2


# Constant bbox for HandRGBD frames (images are always 1280x600).
_HANDRGBD_IMAGE_SIZE = (1280, 600)
_HANDRGBD_CENTERED_BBOX = _center_square_bbox(*_HANDRGBD_IMAGE_SIZE)


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


def load_handrgbd_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    dataset_root = _resolve_root(root)
    rgb_root = dataset_root / "handRGBD" / "rgb_jpg"
    metadata_csv = dataset_root / "handRGBD" / "reference_table.csv"

    empty_cols = ["source", "user_id", "age", "gender", "aspect", "image_path", "bbox"]
    if not metadata_csv.exists():
        return pd.DataFrame(columns=empty_cols)

    raw_df = pd.read_csv(metadata_csv)
    if raw_df.empty or "name" not in raw_df.columns or "user_id" not in raw_df.columns:
        return pd.DataFrame(columns=empty_cols)

    working_df = raw_df.copy()
    working_df = working_df[~working_df["name"].astype(str).str.contains("wall-3", case=False, na=False)]
    
    if "aspect" in working_df.columns:
        working_df["aspect_norm"] = working_df["aspect"].apply(_normalise_label)
    else:
        working_df["aspect_norm"] = pd.NA
    working_df = working_df[working_df["aspect_norm"].notna()]

    def resolve_path(name_val: object) -> Optional[Path]:
        if name_val is None or (isinstance(name_val, float) and pd.isna(name_val)):
            return None
        name_str = str(name_val).strip()
        if not name_str:
            return None
        filename = name_str if name_str.lower().endswith(".jpg") else f"{name_str}.jpg"
        candidate = rgb_root / filename
        return candidate if candidate.is_file() else None

    working_df["image_path"] = working_df["name"].apply(resolve_path)
    working_df = working_df[working_df["image_path"].notna()]

    if "gender" in working_df.columns:
        working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    else:
        working_df["gender_norm"] = None

    if "age" in working_df.columns:
        working_df["age_norm"] = working_df["age"].apply(lambda a: int(a) if pd.notna(a) else pd.NA)
    else:
        working_df["age_norm"] = pd.NA

    working_df["bbox_tuple"] = [tuple(_HANDRGBD_CENTERED_BBOX) for _ in range(len(working_df))]

    df_out = pd.DataFrame(
        {
            "source": "handrgbd",
            "user_id": working_df["user_id"].apply(lambda uid: f"handrgbd_{uid}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"].apply(Path),
            "bbox": working_df["bbox_tuple"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"HandRGBD dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def _limit_users_per_age(df: pd.DataFrame, *, max_users_per_year: int = 15) -> pd.DataFrame:
    """Cap unique users per age, dropping lowest-priority sources first (archive, then primary)."""
    required_cols = {"user_id", "age", "source"}
    if not required_cols.issubset(df.columns):
        return df

    age_known = df[df["age"].notna()].copy()
    if age_known.empty:
        return df

    age_known["age_year"] = age_known["age"].astype(float).round().astype(int)
    priority_map = {"handrgbd": 0, "primary": 1, "archive": 2}
    age_known["priority"] = age_known["source"].map(priority_map).fillna(99).astype(int)

    keep_users: set[str] = set()
    dropped_users = 0
    for _age, group in age_known.groupby("age_year"):
        user_priorities = (
            group.groupby("user_id")["priority"]
            .min()
            .reset_index()
            .sort_values(["priority", "user_id"])
        )
        selected = user_priorities.head(max_users_per_year)["user_id"].tolist()
        keep_users.update(selected)
        dropped_users += max(0, len(user_priorities) - len(selected))

    filtered_known = age_known[age_known["user_id"].isin(keep_users)].drop(columns=["age_year", "priority"])
    if dropped_users:
        print(f"Per-age cap applied ({max_users_per_year} users/year): dropped {dropped_users} users.")

    age_unknown = df[df["age"].isna()]
    return pd.concat([age_unknown, filtered_known], ignore_index=True)


def load_combined_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    primary_df = load_primary_metadata(root=root)
    archive_df = load_archive_metadata(root=root)
    handrgbd_df = load_handrgbd_metadata(root=root)
    combined = pd.concat([primary_df, archive_df, handrgbd_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="image_path")
    combined = _limit_users_per_age(combined, max_users_per_year=15)
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

    # Plot age histogram for a quick sanity check.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping age histogram.")
    else:
        per_user_age = (
            combined.dropna(subset=["age"])
            .groupby("user_id")["age"]
            .first()
            .astype(float)
        )
        if per_user_age.empty:
            print("No age values available; histogram skipped.")
        else:
            # Build stacked bar chart by dataset source.
            source_for_user = (
                combined.dropna(subset=["age"])
                .groupby("user_id")["source"]
                .first()
            )
            colour_map = {"primary": "#4c72b0", "archive": "#dd8452", "handrgbd": "#55a868"}
            unique_sources = source_for_user.unique()

            min_age = float(np.floor(per_user_age.min()))
            max_age = float(np.ceil(per_user_age.max()))
            if min_age == max_age:
                bin_edges = np.array([min_age - 0.5, max_age + 0.5])
            else:
                bin_edges = np.arange(min_age - 0.5, max_age + 1.5, 1.0)  # one bin per year
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bar_width = 1.0

            plt.figure(figsize=(9, 5))
            cumulative = np.zeros_like(bin_centers, dtype=float)
            for src in unique_sources:
                mask = source_for_user[source_for_user == src].index
                ages = per_user_age.loc[per_user_age.index.isin(mask)]
                if ages.empty:
                    continue
                counts, _ = np.histogram(ages, bins=bin_edges)
                plt.bar(
                    bin_centers,
                    counts,
                    width=bar_width * 0.9,
                    bottom=cumulative,
                    color=colour_map.get(src, None),
                    edgecolor="black",
                    alpha=0.85,
                    label=src,
                )
                cumulative = cumulative + counts
            if len(unique_sources) > 1:
                plt.legend(title="Source")

            plt.xlabel("Age")
            plt.ylabel("User count")
            plt.title("Hand datasets age distribution (per user)")
            plt.tight_layout()
            output_path = Path("age_histogram_users.png")
            plt.savefig(output_path)
            print(f"Saved age histogram to {output_path}")
            plt.show()
