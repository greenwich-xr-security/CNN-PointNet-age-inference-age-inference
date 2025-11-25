import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Iterable, Optional, Tuple
from collections import defaultdict


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
        for thr in (18.0, 25.0):
            plt.axvline(thr, color="black", linestyle=":", linewidth=1)
            plt.axhline(thr, color="black", linestyle=":", linewidth=1)
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
    def make_square_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Compute a square bounding box centered on the given bbox."""
        if bbox is None:
            raise ValueError("bbox must not be None")
        if len(bbox) != 4:
            raise ValueError("bbox must be a tuple (xmin, ymin, xmax, ymax)")

        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox has invalid coordinates: {bbox}")

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        side = max(bw, bh)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        sx1 = int(np.floor(cx - side / 2.0))
        sy1 = int(np.floor(cy - side / 2.0))
        sx2 = sx1 + side
        sy2 = sy1 + side
        return sx1, sy1, sx2, sy2

    # ------------------------------------------------------------------ #
    @staticmethod
    def save_regression_scatter(
        targets: Iterable[float],
        predictions: Iterable[float],
        *,
        save_path,
        title: Optional[str] = None,
        axis_limits: Optional[Tuple[float, float]] = None,
        point_size: int = 20,
        alpha: float = 0.6,
    ) -> bool:
        """Save a regression scatter plot with 3x3 colouring based on age brackets (<18, 18-25, >25)."""
        targets_arr = np.asarray(list(targets), dtype=float)
        preds_arr = np.asarray(list(predictions), dtype=float)
        if targets_arr.size == 0:
            print("save_regression_scatter: no data to plot.")
            return False

        if axis_limits is not None:
            axis_min, axis_max = axis_limits
        else:
            min_val = float(np.min([targets_arr.min(), preds_arr.min()]))
            max_val = float(np.max([targets_arr.max(), preds_arr.max()]))
            padding = max(1.0, 0.05 * (max_val - min_val))
            axis_min = min_val - padding
            axis_max = max_val + padding

        def _bin(x: float) -> int:
            if x < 18.0:
                return 0
            if x < 25.0:
                return 1
            return 2

        colors = []
        for t, p in zip(targets_arr, preds_arr):
            tb = _bin(t)
            pb = _bin(p)
            if tb == pb:
                colors.append("green")  # on-diagonal
            elif (tb == 0 and pb == 2) or (tb == 2 and pb == 0):
                colors.append("red")  # opposite corners
            else:
                colors.append("gold")  # adjacent/off-diagonal

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(targets_arr, preds_arr, s=point_size, alpha=alpha, edgecolors="none", c=colors)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1)
        for thr in (18.0, 25.0):
            ax.axvline(thr, color="black", linestyle=":", linewidth=1)
            ax.axhline(thr, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("True Age")
        ax.set_ylabel("Predicted Age")
        if title:
            ax.set_title(title)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return True

    @staticmethod
    def save_per_user_boxplot(
        user_ids: Iterable,
        targets: Iterable[float],
        predictions: Iterable[float],
        *,
        save_path,
        title: Optional[str] = None,
    ) -> bool:
        """Save a box plot of predicted age distributions per user positioned at true age."""
        user_ids = list(user_ids)
        targets_arr = np.asarray(list(targets), dtype=float)
        preds_arr = np.asarray(list(predictions), dtype=float)
        if len(user_ids) == 0 or targets_arr.size == 0 or preds_arr.size == 0:
            print("save_per_user_boxplot: no data to plot.")
            return False
        if not (len(user_ids) == targets_arr.size == preds_arr.size):
            print("save_per_user_boxplot: input lengths mismatch.")
            return False

        per_user_preds = defaultdict(list)
        per_user_targets = defaultdict(list)
        for uid, tgt, pred in zip(user_ids, targets_arr, preds_arr):
            per_user_preds[uid].append(float(pred))
            per_user_targets[uid].append(float(tgt))

        positions = []
        box_data = []
        colours = []

        def _bin(x: float) -> int:
            if x < 18.0:
                return 0
            if x < 25.0:
                return 1
            return 2
        for uid, preds in per_user_preds.items():
            if not preds:
                continue
            tgt_mean = float(np.mean(per_user_targets[uid])) if per_user_targets[uid] else 0.0
            positions.append(tgt_mean)
            box_data.append(preds)
            tb = _bin(tgt_mean)
            pb = _bin(np.median(preds))
            if tb == pb:
                colours.append("green")
            elif (tb == 0 and pb == 2) or (tb == 2 and pb == 0):
                colours.append("red")
            else:
                colours.append("gold")

        if not box_data:
            print("save_per_user_boxplot: no per-user predictions to plot.")
            return False

        fig, ax = plt.subplots(figsize=(6, 6))
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor="white", alpha=0.6),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="#4575b4"),
            capprops=dict(color="#4575b4"),
            flierprops=dict(marker=".", markersize=2, markerfacecolor="#313695", alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], colours):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
        medians = [np.median(b) if len(b) > 0 else 0.0 for b in box_data]
        # median marker matches box colour
        ax.scatter(positions, medians, color=[c for c in colours], s=8)
        ax.set_xlabel("True age (per user mean)")
        ax.set_ylabel("Predicted age (distribution per user)")
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.2)
        # axis colour kept default; ticks handled below

        axis_min = 0.0
        axis_max = 60.0
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1, label="Ideal")
        for thr in (18.0, 25.0):
            ax.axvline(thr, color="black", linestyle=":", linewidth=1)
            ax.axhline(thr, color="black", linestyle=":", linewidth=1)

        tick_values = [10, 20, 30, 40, 50, 60]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(v) for v in tick_values])
        ax.set_yticks(tick_values)
        ax.set_yticklabels([str(v) for v in tick_values])
        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, length=4)
        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_loss_history(
        history: Iterable[dict],
        *,
        save_path=None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> Optional[Path]:
        """Plot training/validation MAE and RMSE curves over epochs.

        Args:
            history: Iterable of dicts with keys: epoch, train_mae, val_mae, train_rmse, val_rmse.
            save_path: Optional path to save the plot. Creates parents as needed.
            show: Whether to display the plot (default True).
            title: Optional plot title.
        Returns:
            Path to the saved plot if saved, otherwise None.
        """
        entries = list(history)
        if not entries:
            print("plot_loss_history: no history entries provided.")
            return None

        epochs = [entry["epoch"] for entry in entries]
        train_mae = [entry["train_mae"] for entry in entries]
        val_mae = [entry["val_mae"] for entry in entries]
        train_rmse = [entry["train_rmse"] for entry in entries]
        val_rmse = [entry["val_rmse"] for entry in entries]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_mae, label="Train MAE", color="#1f77b4")
        ax.plot(epochs, val_mae, label="Val MAE", color="#ff7f0e")
        ax.plot(epochs, train_rmse, label="Train RMSE", color="#2ca02c")
        ax.plot(epochs, val_rmse, label="Val RMSE", color="#d62728")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        if title:
            ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        saved_path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            saved_path = save_path

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_roc_curve(
        fprs,
        tprs,
        *,
        thresholds,
        save_path,
        title: Optional[str] = None,
        auc_value: Optional[float] = None,
        show: bool = False,
        highlight_points: Optional[list] = None,
        highlight_labels: Optional[list] = None,
    ) -> Optional[Path]:
        """Plot an ROC-style curve with thresholds encoded by colour."""
        fprs_arr = np.asarray(list(fprs), dtype=float)
        tprs_arr = np.asarray(list(tprs), dtype=float)
        thresholds_arr = np.asarray(list(thresholds), dtype=float)
        if fprs_arr.size == 0 or tprs_arr.size == 0:
            print("plot_roc_curve: no data to plot.")
            return None

        order = np.argsort(fprs_arr)
        ordered_fprs = fprs_arr[order]
        ordered_tprs = tprs_arr[order]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
        label = "ROC"
        if auc_value is not None:
            label += f" (AUC={auc_value:.3f})"
        ax.plot(ordered_fprs, ordered_tprs, color="#1f77b4", label=label)
        scatter = ax.scatter(
            fprs_arr,
            tprs_arr,
            c=thresholds_arr,
            cmap="viridis",
            s=35,
            edgecolors="none",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Threshold tau", rotation=270, labelpad=15)
        ax.set_xlabel("FPR (undesired risk)")
        ax.set_ylabel("TPR (desired usability)")
        if title:
            ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.legend(loc="lower right")
        if highlight_points:
            hl = highlight_labels or [None] * len(highlight_points)
            for (x, y), lbl in zip(highlight_points, hl):
                ax.scatter([x], [y], marker="s", color="black", s=45, zorder=5)
                if lbl:
                    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8, color="black")
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return save_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_to_max(img, max_dim):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img, scale
