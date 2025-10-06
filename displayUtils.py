import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Iterable, Optional, Tuple


class DisplayUtils:
    # ------------------------------------------------------------------ #
    # connections (unchanged)
    _CONNECTIONS = (
        (0, 1),  (1, 2),  (2, 3),  (3, 4),        # thumb
        (0, 5),  (5, 6),  (6, 7),  (7, 8),        # index
        (0, 9),  (9, 10), (10, 11), (11, 12),     # middle
        (0, 13), (13, 14), (14, 15), (15, 16),    # ring
        (0, 17), (17, 18), (18, 19), (19, 20)     # little
    )

    # ------------------------------------------------------------------ #
    @staticmethod
    def display_with_coords(
        img,
        landmarks,
        *,
        handedness=None,
        palm_side=None,
        colour=(0, 255, 0),
        max_dim=1080,
    ) -> None:
        """Draw hand landmarks (normalised to the FULL image) on *img*."""
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (21, 3):
            raise ValueError("landmarks must have shape (21, 3)")

        work_img = img.copy()
        work_img, scale = DisplayUtils._resize_to_max(work_img, max_dim)
        h_img, w_img = work_img.shape[:2]

        # Points + indices
        for idx, (lx, ly, _lz) in enumerate(landmarks):
            x_px = int(np.clip(lx, 0.0, 1.0) * w_img)
            y_px = int(np.clip(ly, 0.0, 1.0) * h_img)
            cv2.circle(work_img, (x_px, y_px), 4, colour, -1)
            cv2.putText(
                work_img,
                f"{idx}:({x_px},{y_px})",
                (x_px + 5, y_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

        # Skeleton
        for a, b in DisplayUtils._CONNECTIONS:
            xa = int(np.clip(landmarks[a, 0], 0.0, 1.0) * w_img)
            ya = int(np.clip(landmarks[a, 1], 0.0, 1.0) * h_img)
            xb = int(np.clip(landmarks[b, 0], 0.0, 1.0) * w_img)
            yb = int(np.clip(landmarks[b, 1], 0.0, 1.0) * h_img)
            cv2.line(work_img, (xa, ya), (xb, yb), colour, 2)

        # Orientation overlay
        orientation_text = " ".join(t for t in (handedness, palm_side) if t).strip()
        if orientation_text:
            cv2.putText(
                work_img,
                orientation_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        title = "Hand Landmarks (image-normalised)"
        if orientation_text:
            title += f" | {orientation_text}"
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.show()

    @staticmethod
    def display(
        img,
        *,
        max_dim=1080,
        title: Optional[str] = None,
    ) -> None:
        """Display a BGR image using Matplotlib after optional resizing."""
        work_img = img.copy()
        work_img, _ = DisplayUtils._resize_to_max(work_img, max_dim)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def display_with_bbox(
        img,
        bbox: Tuple[int, int, int, int],
        *,
        max_dim: int = 1080,
        colour: Tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
        title: Optional[str] = None,
    ) -> None:
        """Display an image with a single bounding box overlay."""
        if bbox is None:
            raise ValueError("bbox must not be None")
        if len(bbox) != 4:
            raise ValueError("bbox must be a tuple (xmin, ymin, xmax, ymax)")

        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        if xmax < xmin or ymax < ymin:
            raise ValueError(f"bbox has invalid coordinates: {bbox}")

        work_img = img.copy()
        work_img, scale = DisplayUtils._resize_to_max(work_img, max_dim)
        xmin_s = int(round(xmin * scale))
        ymin_s = int(round(ymin * scale))
        xmax_s = int(round(xmax * scale))
        ymax_s = int(round(ymax * scale))
        cv2.rectangle(work_img, (xmin_s, ymin_s), (xmax_s, ymax_s), colour, thickness)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def display_regression_scatter(
        targets: Iterable[float],
        predictions: Iterable[float],
        *,
        title: Optional[str] = None,
        point_size: int = 20,
        alpha: float = 0.6,
    ) -> None:
        """Scatter plot of targets vs predictions with equal axes."""
        targets_arr = np.asarray(list(targets), dtype=float)
        preds_arr = np.asarray(list(predictions), dtype=float)
        if targets_arr.size == 0:
            print("display_regression_scatter: no data to plot.")
            return

        min_val = float(np.min([targets_arr.min(), preds_arr.min()]))
        max_val = float(np.max([targets_arr.max(), preds_arr.max()]))
        padding = max(1.0, 0.05 * (max_val - min_val))
        axis_min = min_val - padding
        axis_max = max_val + padding

        plt.figure(figsize=(6, 6))
        plt.scatter(targets_arr, preds_arr, s=point_size, alpha=alpha, edgecolors='none')
        plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=1)
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        if title:
            plt.title(title)
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_to_max(img, max_dim):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img, scale
