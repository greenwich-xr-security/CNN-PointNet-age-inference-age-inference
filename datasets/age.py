"""Dataset utilities for age regression tasks."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from displayUtils import DisplayUtils

DEFAULT_NUM_POINTS = 2048


class AgeDataset(Dataset):
    """Returns (image, pointcloud, age, user_id) tuples for multimodal training."""

    def __init__(
        self,
        records: pd.DataFrame,
        *,
        use_rgb: bool = True,
        use_pointcloud: bool = False,
        transform=None,
        num_points: int = DEFAULT_NUM_POINTS,
        pc_jitter_std: float = 0.0,
    ):
        self.records = records.reset_index(drop=True)
        self.transform = transform
        self.use_rgb = use_rgb
        self.use_pointcloud = use_pointcloud
        self.num_points = num_points
        self.pc_jitter_std = pc_jitter_std

    def __len__(self):
        return len(self.records)

    def _load_image(self, row):
        image_path: Path = row["image_path"]
        image = Image.open(image_path).convert("RGB")

        bbox = row.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox]
                if xmax > xmin and ymax > ymin:
                    w, h = image.size
                    sq_xmin, sq_ymin, sq_xmax, sq_ymax = DisplayUtils.make_square_bbox(
                        (xmin, ymin, xmax, ymax)
                    )

                    pad_left = max(0, -sq_xmin)
                    pad_top = max(0, -sq_ymin)
                    pad_right = max(0, sq_xmax - w)
                    pad_bottom = max(0, sq_ymax - h)

                    if pad_left or pad_top or pad_right or pad_bottom:
                        image = ImageOps.expand(
                            image,
                            border=(pad_left, pad_top, pad_right, pad_bottom),
                            fill=(0, 0, 0),
                        )
                        sq_xmin += pad_left
                        sq_xmax += pad_left
                        sq_ymin += pad_top
                        sq_ymax += pad_top

                    sq_xmin = max(0, sq_xmin)
                    sq_ymin = max(0, sq_ymin)
                    sq_xmax = max(sq_xmin + 1, min(image.size[0], sq_xmax))
                    sq_ymax = max(sq_ymin + 1, min(image.size[1], sq_ymax))
                    image = image.crop((sq_xmin, sq_ymin, sq_xmax, sq_ymax))
            except Exception:
                pass
        if self.transform:
            image = self.transform(image)
        return image

    def _load_points(self, row) -> torch.Tensor:
        xyz_path = row.get("xyz_path")
        if xyz_path is None or pd.isna(xyz_path):
            raise RuntimeError("Point cloud requested but xyz_path missing.")
        xyz_path = Path(xyz_path)
        if not xyz_path.is_file():
            raise RuntimeError(f"Point cloud file not found: {xyz_path}")
        coords = np.load(xyz_path, allow_pickle=False)
        if coords.ndim == 3 and coords.shape[-1] >= 3:
            coords = coords.reshape(-1, coords.shape[-1])
        if coords.ndim != 2 or coords.shape[1] < 3:
            raise RuntimeError(f"Point cloud has unexpected shape {coords.shape}")
        coords = coords[:, :3].astype(np.float32)
        finite_mask = np.isfinite(coords).all(axis=1)
        coords = coords[finite_mask]
        if coords.shape[0] == 0:
            raise RuntimeError(f"Point cloud empty: {xyz_path}")

        n_points = coords.shape[0]
        target = max(1, int(self.num_points))
        if n_points >= target:
            idx = np.random.choice(n_points, target, replace=False)
        else:
            idx = np.random.choice(n_points, target, replace=True)
        coords = coords[idx]

        coords = coords - np.mean(coords, axis=0, keepdims=True)
        norms = np.linalg.norm(coords, axis=1, keepdims=True)
        max_norm = float(np.max(norms)) if norms.size else 1.0
        if max_norm > 0:
            coords = coords / max_norm
        if self.pc_jitter_std > 0:
            coords = coords + np.random.normal(scale=self.pc_jitter_std, size=coords.shape).astype(np.float32)
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(coords)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        age = float(row["age"])
        user_id = row.get("user_id")
        image = self._load_image(row) if self.use_rgb else None
        points = self._load_points(row) if self.use_pointcloud else None
        return image, points, torch.tensor(age, dtype=torch.float32), user_id
