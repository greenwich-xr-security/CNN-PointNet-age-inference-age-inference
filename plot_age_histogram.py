#!/usr/bin/env python
"""Plot age distribution for combined hand datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hands_dataset import (
    load_archive_metadata,
    load_primary_metadata,
)


def _load_user_ages(root: str | None) -> pd.Series:
    """Return a Series of user ages aggregated across both datasets."""
    primary_df = load_primary_metadata(root=root)
    archive_df = load_archive_metadata(root=root)
    combined = pd.concat([primary_df, archive_df], ignore_index=True)

    if combined.empty or "age" not in combined.columns:
        return pd.Series(dtype=float)

    combined["age"] = pd.to_numeric(combined["age"], errors="coerce")
    combined = combined.dropna(subset=["age", "user_id"])

    # Assume age per user is consistent; keep the first occurrence.
    user_ages = (
        combined.sort_values("user_id")
        .drop_duplicates(subset="user_id")
        .set_index("user_id")["age"]
        .astype(float)
    )
    return user_ages


def plot_age_histogram(
    user_ages: pd.Series,
    *,
    bins: int,
    show: bool,
    save_path: Path | None,
) -> None:
    if user_ages.empty:
        print("No age data found; nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(user_ages.values, bins=bins, color="#1f77b4", alpha=0.75, edgecolor="black")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of users")
    ax.set_title(f"User Age Distribution (n={len(user_ages)})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved histogram to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram of user ages across primary and archive datasets."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Custom datasets root (defaults to environment or hands_dataset default).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins (default: 20).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the histogram image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable on-screen display (useful when only saving to file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_ages = _load_user_ages(args.data_root)
    print(f"Total unique users with age information: {len(user_ages)}")
    if user_ages.empty:
        return

    mean_age = float(np.mean(user_ages))
    median_age = float(np.median(user_ages))
    print(f"Mean age: {mean_age:.2f} | Median age: {median_age:.2f}")

    save_path = Path(args.save) if args.save else None
    plot_age_histogram(
        user_ages,
        bins=max(1, args.bins),
        show=not args.no_show,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
