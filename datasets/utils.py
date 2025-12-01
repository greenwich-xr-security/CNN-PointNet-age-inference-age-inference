"""General dataset utilities shared by training scripts."""

from typing import Optional

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
    num_bins: int = 4,
    target_bin_size: Optional[int] = None,
    return_bin_info: bool = False,
) -> tuple:
    """
    Stratify users into quantile age bins (equal user counts) then split per bin.

    If bins are too small to allow both train and test samples, the function
    progressively reduces the number of bins; if no stratified split is
    possible, it falls back to an adult/minor split, then to an unstratified
    split.

    Returns train_ids, test_ids, and optionally a bin summary (when
    return_bin_info=True).
    """
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

    per_user["age_int"] = per_user["mean_age"].round().astype(int)
    user_ids = per_user["user_id"].to_numpy()
    user_count = len(user_ids)

    def _build_bins(max_bins: int):
        for bins in range(max_bins, 0, -1):
            bin_labels, bin_edges = pd.qcut(
                per_user["age_int"],
                q=bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            if bin_labels.isna().any():
                continue
            bin_labels = bin_labels.astype(int)
            actual_bins = int(bin_labels.max()) + 1
            if actual_bins < 2:
                continue
            counts = bin_labels.value_counts().sort_index().to_numpy()
            n_test = np.ceil(counts * test_size).astype(int)
            n_train = counts - n_test
            if np.all(counts > 0) and np.all(n_test >= 1) and np.all(n_train >= 1):
                return bin_labels.to_numpy(), bin_edges
        return None, None

    desired_bins = max(num_bins, 2)
    if target_bin_size and target_bin_size > 0:
        approx_bins = max(2, int(round(user_count / target_bin_size)))
        desired_bins = max(desired_bins, approx_bins)

    strat_labels, bin_edges = _build_bins(desired_bins)
    stratify = None
    if strat_labels is None:
        # Fallback to adult/minor split if possible
        binary_labels = (per_user["mean_age"].to_numpy() >= adult_threshold).astype(int)
        unique_labels, label_counts = np.unique(binary_labels, return_counts=True)
        if unique_labels.size > 1:
            n_test = np.ceil(label_counts * test_size).astype(int)
            n_train = label_counts - n_test
            if np.all(n_test >= 1) and np.all(n_train >= 1):
                strat_labels = binary_labels
                min_age = float(per_user["mean_age"].min())
                max_age = float(per_user["mean_age"].max())
                lower = min(min_age, adult_threshold)
                upper = max(max_age, adult_threshold)
                bin_edges = np.array([lower, adult_threshold, upper])
    if strat_labels is not None:
        stratify = strat_labels

    train_ids, test_ids = train_test_split(
        user_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    if not return_bin_info:
        return train_ids, test_ids

    bin_summary: list[dict] = []
    if strat_labels is not None and bin_edges is not None:
        per_user = per_user.assign(bin=strat_labels)
        train_set = set(train_ids)
        test_set = set(test_ids)
        edge_list = [int(round(x)) for x in bin_edges]
        for bin_idx in range(int(per_user["bin"].max()) + 1):
            bin_rows = per_user[per_user["bin"] == bin_idx]
            bin_users = bin_rows["user_id"]
            bin_ages = bin_rows["age_int"].to_numpy()
            user_count = len(bin_users)
            train_count = sum(uid in train_set for uid in bin_users)
            test_count = user_count - train_count
            bin_summary.append(
                {
                    "bin": int(bin_idx),
                    "age_min": float(edge_list[bin_idx]),
                    "age_max": float(edge_list[bin_idx + 1]),
                    "age_min_obs": float(np.min(bin_ages)) if bin_ages.size else None,
                    "age_max_obs": float(np.max(bin_ages)) if bin_ages.size else None,
                    "users": int(user_count),
                    "train_users": int(train_count),
                    "test_users": int(test_count),
                }
            )
    return train_ids, test_ids, bin_summary


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
