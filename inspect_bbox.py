"""Visual inspection tool for bounding boxes stored in the hand datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple
import random

import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from displayUtils import DisplayUtils
from hands_dataset import (
    load_archive_metadata,
    load_combined_metadata,
    load_handrgbd_metadata,
    load_primary_metadata,
)


def _iter_rows(
    bbox_source: Iterable,
    start: int,
    limit: int | None,
) -> Iterable[Tuple[int, object]]:
    df = bbox_source
    if start:
        df = df.iloc[start:]
    if limit is not None:
        df = df.iloc[:limit]
    return df.iterrows()


def _load_metadata(source: str):
    if source == "primary":
        return load_primary_metadata()
    if source == "archive":
        return load_archive_metadata()
    if source == "handrgbd":
        return load_handrgbd_metadata()
    return load_combined_metadata()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect bounding boxes visually.")
    parser.add_argument(
        "--source",
        choices=("primary", "archive", "handrgbd", "combined"),
        default="combined",
        help="Dataset partition to inspect.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index inside the chosen dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to display.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically advance without waiting for keyboard input.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1080,
        help="Resize the longer image edge to this size before display.",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="Filter to samples whose aspect (e.g., dorsal) matches this value.",
    )
    return parser.parse_args()


def _show_image_with_buttons(img, max_dim: int, title: str | None = None) -> str:
    """Display image with Next/Quit buttons; return the selected action."""
    work = img.copy()
    work, _ = DisplayUtils._resize_to_max(work, max_dim)
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(rgb)
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)

    action = {"value": "next"}

    button_height = 0.08
    button_width = 0.2
    spacing = 0.05
    bottom = 0.05
    next_left = 0.5 - button_width - spacing / 2
    quit_left = 0.5 + spacing / 2

    ax_next = fig.add_axes([next_left, bottom, button_width, button_height])
    ax_quit = fig.add_axes([quit_left, bottom, button_width, button_height])

    btn_next = Button(ax_next, "Next")
    btn_quit = Button(ax_quit, "Quit", color="#d9534f", hovercolor="#c9302c")

    def _on_next(_event):
        action["value"] = "next"
        plt.close(fig)

    def _on_quit(_event):
        action["value"] = "quit"
        plt.close(fig)

    btn_next.on_clicked(_on_next)
    btn_quit.on_clicked(_on_quit)

    plt.show()
    return action["value"]


def main() -> None:
    args = _parse_args()
    df = _load_metadata(args.source)

    if args.aspect:
        aspect_col = "aspect"
        if aspect_col not in df.columns:
            print(f"Dataset is missing '{aspect_col}' column; cannot filter on aspect.")
            return
        target_aspect = args.aspect.strip().lower()
        df = df[
            df[aspect_col]
            .astype(str)
            .str.lower()
            .str.contains(target_aspect, na=False)
        ]

    if df.empty:
        aspect_msg = f" and aspect '{args.aspect}'" if args.aspect else ""
        print(f"No samples found for the chosen configuration{aspect_msg}.")
        return

    # Iterate in random order instead of sequential
    indices = list(range(len(df)))
    random.shuffle(indices)
    if args.start:
        indices = indices[args.start:]
    if args.limit is not None:
        indices = indices[: args.limit]

    for idx in indices:
        row = df.iloc[idx]
        bbox = row.get("bbox")
        if bbox is None:
            print(f"[inspect] Missing bbox for index {idx}, skipping.")
            continue

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_tuple = tuple(int(v) for v in bbox)
        else:
            print(f"[inspect] Invalid bbox format for index {idx}: {bbox!r}")
            continue

        image_path = Path(row["image_path"])
        if not image_path.is_file():
            print(f"[inspect] Image not found at {image_path}, skipping.")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[inspect] Failed to read {image_path}, skipping.")
            continue

        # Draw original bbox (yellow) and equally padded square bbox (magenta)
        work = img.copy()
        h, w = work.shape[:2]
        x1, y1, x2, y2 = bbox_tuple
        # original bbox
        cv2.rectangle(work, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        try:
            sx1, sy1, sx2, sy2 = DisplayUtils.make_square_bbox(bbox_tuple)
        except ValueError as exc:
            print(f"[inspect] {exc} for index {idx}, skipping.")
            continue
        # Clamp for display (padding beyond edges is not visible on original image)
        sx1_d = max(0, sx1)
        sy1_d = max(0, sy1)
        sx2_d = min(w - 1, sx2)
        sy2_d = min(h - 1, sy2)
        if sx2_d > sx1_d and sy2_d > sy1_d:
            cv2.rectangle(work, (sx1_d, sy1_d), (sx2_d, sy2_d), (255, 0, 255), 2)

        title = f"{idx} - {row.get('source', '?')} - {row.get('aspect', '?')} - {image_path.name}"
        if args.auto:
            DisplayUtils.display(
                work,
                max_dim=args.max_dim,
                title=title,
            )
        else:
            action = _show_image_with_buttons(work, args.max_dim, title)
            if action == "quit":
                break


if __name__ == "__main__":
    main()
