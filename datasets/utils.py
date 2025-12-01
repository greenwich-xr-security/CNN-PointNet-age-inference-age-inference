"""General dataset utilities shared by training scripts."""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def filter_metadata(df: pd.DataFrame, *, require_xyz: bool = False) -> pd.DataFrame:
    """Filter to dorsal samples with known ages (and xyz when required)."""
    df = df[df["aspect"].str.contains("dorsal", case=False, na=False)]
    df = df[df["age"].notna()]
    if require_xyz:
        df = df[df["xyz_path"].notna()]
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


def multimodal_collate(batch):
    """Collate batch of (image, points, age, user_id) tuples."""
    images = [b[0] for b in batch]
    points = [b[1] for b in batch]
    ages = torch.stack([b[2] for b in batch])
    user_ids = [b[3] for b in batch]

    image_tensor = None
    point_tensor = None

    if images and images[0] is not None:
        image_tensor = torch.stack(images)
    if points and points[0] is not None:
        point_tensor = torch.stack(points)

    return image_tensor, point_tensor, ages, user_ids
