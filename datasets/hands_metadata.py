"""Utility script for loading and normalising the HandRGBD dataset.

The module exposes helper functions to load the metadata into a Pandas
DataFrame as well as a ``HandsDataset`` class that can be re-used by training
or inference scripts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from utils import stratified_user_split

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
# BBox helpers

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


# ---------------------------------------------------------------------------
# Metadata loaders

def load_handrgbd_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    dataset_root = _resolve_root(root)
    hand_root = dataset_root / "handRGBD"
    rgb_root = hand_root / "rgb_jpg"
    if not rgb_root.exists():
        alt_root = hand_root / "rgb"
        if alt_root.exists():
            rgb_root = alt_root
    xyz_root = hand_root / "xyz_npy"
    if not rgb_root.exists():
        print(f"[handRGBD] RGB folder not found (tried 'rgb_jpg' and 'rgb' under {hand_root})")
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path", "bbox"])
    metadata_csv = dataset_root / "handRGBD" / "reference_table.csv"

    empty_cols = ["source", "user_id", "age", "gender", "aspect", "image_path", "bbox", "xyz_path"]
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
        candidates = []
        if any(name_str.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
            candidates.append(rgb_root / name_str)
        else:
            candidates.append(rgb_root / f"{name_str}.png")
            candidates.append(rgb_root / f"{name_str}.jpg")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def resolve_xyz(name_val: object) -> Optional[Path]:
        if not xyz_root.exists():
            return None
        if name_val is None or (isinstance(name_val, float) and pd.isna(name_val)):
            return None
        base_name = str(name_val).strip()
        if not base_name:
            return None
        base_name = Path(base_name).stem  # drop extension if present
        candidate = xyz_root / f"{base_name}.npy"
        if candidate.is_file():
            return candidate
        return None

    working_df["image_path"] = working_df["name"].apply(resolve_path)
    working_df = working_df[working_df["image_path"].notna()]
    working_df["xyz_path"] = working_df["name"].apply(resolve_xyz)

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
            "xyz_path": working_df["xyz_path"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"HandRGBD dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_combined_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    handrgbd_df = load_handrgbd_metadata(root=root)
    combined = pd.concat([handrgbd_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="image_path")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset class

class HandsDataset(Dataset):
    """Dataset that yields an image tensor, its label, and metadata (optionally XYZ)."""

    def __init__(self, records: pd.DataFrame, transform=None, *, load_xyz: bool = False):
        self.records = records.reset_index(drop=True)
        self.transform = transform
        self.load_xyz = load_xyz

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
        xyz_path_val = row.get("xyz_path") if "xyz_path" in row else None
        xyz_path = None
        xyz = None
        if pd.notna(xyz_path_val) and xyz_path_val:
            xyz_path = Path(xyz_path_val)
        if self.load_xyz:
            if xyz_path is None or not xyz_path.is_file():
                raise RuntimeError(f"XYZ file not available for {image_path}")
            try:
                xyz = np.load(xyz_path, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to load XYZ data {xyz_path}") from exc

        metadata = {
            "user_id": row["user_id"],
            "age": row["age"],
            "gender": row["gender"],
            "source": row["source"],
            "image_path": image_path,
            "xyz_path": xyz_path,
            "xyz": xyz,
        }
        return image, label, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect HandRGBD dataset metadata.")
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
    print(f"HandRGBD samples: {len(combined)}")
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
            bin_info = None
            strat_df = pd.DataFrame({"user_id": per_user_age.index, "age": per_user_age.values})
            try:
                _, _, bin_info = stratified_user_split(
                    strat_df,
                    test_size=0.2,
                    random_state=42,
                    num_bins=13,
                    target_bin_size=30,
                    return_bin_info=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Stratified binning failed: {exc}")

            # Build bar chart by dataset source.
            source_for_user = (
                combined.dropna(subset=["age"])
                .groupby("user_id")["source"]
                .first()
            )
            colour_map = {"handrgbd": "#55a868"}
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

            if bin_info:
                edge_candidates = set()
                for b in bin_info:
                    left = b.get("age_min_obs", b["age_min"])
                    right = b.get("age_max_obs", b["age_max"])
                    edge_candidates.add(left - 0.5)
                    edge_candidates.add(right + 0.5)
                for edge in sorted(edge_candidates):
                    plt.axvline(edge, color="#c44e52", linestyle="--", linewidth=1.2, alpha=0.7)
                y_top = plt.ylim()[1]
                for b in bin_info:
                    left = b.get("age_min_obs", b["age_min"])
                    right = b.get("age_max_obs", b["age_max"])
                    mid = 0.5 * (left + right)
                    plt.text(
                        mid,
                        y_top * 0.9,
                        f"B{b['bin']}: {b['users']}u",
                        ha="center",
                        va="top",
                        fontsize=8,
                        color="#c44e52",
                    )

            plt.xlabel("Age")
            plt.ylabel("User count")
            plt.title("HandRGBD age distribution (per user)")
            plt.tight_layout()
            output_path = Path("age_histogram_users.png")
            plt.savefig(output_path)
            print(f"Saved age histogram to {output_path}")
            if bin_info:
                print("Stratified bin summary (integer ages; right-closed bins shown with observed span):")
                for b in bin_info:
                    obs_min = b.get("age_min_obs")
                    obs_max = b.get("age_max_obs")
                    if obs_min is not None and obs_min.is_integer():
                        obs_min = int(obs_min)
                    if obs_max is not None and obs_max.is_integer():
                        obs_max = int(obs_max)
                    observed_label = ""
                    if obs_min is not None and obs_max is not None:
                        observed_label = f" | observed ages {obs_min}-{obs_max}"
                    label = observed_label.replace(" | ", "").replace("observed ages ", "")
                    pretty = label if label else f"{int(b['age_min'])}-{int(b['age_max'])}"
                    print(
                        f"  Bin {b['bin']}: {pretty} | users={b['users']} train={b['train_users']} test={b['test_users']}"
                    )
            plt.show()

            # Preview one XYZ npy as a 3D scatter if available.
            xyz_candidates = combined.dropna(subset=["xyz_path"])
            if xyz_candidates.empty:
                print("No XYZ files available; skipping 3D preview.")
            else:
                sample_xyz_path = Path(xyz_candidates.iloc[0]["xyz_path"])
                if not sample_xyz_path.is_file():
                    print(f"XYZ file missing on disk: {sample_xyz_path}")
                else:
                    try:
                        coords = np.load(sample_xyz_path, allow_pickle=False)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to load XYZ sample {sample_xyz_path}: {exc}")
                    else:
                        if coords.ndim == 3 and coords.shape[-1] >= 3:
                            coords = coords.reshape(-1, coords.shape[-1])
                        if coords.ndim != 2 or coords.shape[1] < 3:
                            print(f"XYZ sample has unexpected shape {coords.shape}; skipping 3D preview.")
                        else:
                            coords = coords[:, :3]
                            if len(coords) > 5000:
                                step = max(1, len(coords) // 5000)
                                coords = coords[::step]
                            fig = plt.figure(figsize=(6, 5))
                            ax = fig.add_subplot(111, projection="3d")
                            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2, alpha=0.8)
                            ax.set_xlabel("X")
                            ax.set_ylabel("Y")
                            ax.set_zlabel("Z")
                            ax.set_title(f"HandRGBD XYZ sample\n{sample_xyz_path.name}")
                            plt.tight_layout()
                            plt.show()
