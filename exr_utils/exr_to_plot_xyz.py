#!/usr/bin/env python3
"""
plot_xyz_from_exr_with_passthrough.py
-------------------------------------
Reads an EXR (depth+XYZ) and a corresponding passthrough PNG (RGB).
Builds a colored point cloud and saves it as PLY.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import exr_utils as eu
import imageio.v2 as imageio   # for reading PNG/JPG


# -------- Config (direct import) --------
from config import DEPTH_PATH, RGB_PATH, PLYRGB_PATH, STRIDE, DEPTH_MIN, DEPTH_MAX, NEGATE_Z, LIMIT, SELECTED_ITEMS


def plot_xyz_with_passthrough(exr_path: Path,
                              png_path: Path,
                              stride: int = 4,
                              depth_min: float | None = None,
                              depth_max: float | None = None,
                              negate_z: bool = False,
                              limit: int | None = None,
                              ply_out=None, show_plot: bool = False) -> None:
    # --- Read EXR channels ---
    chans = eu.read_channels(str(exr_path), channels=("R", "G", "B", "A"))
    X = np.asarray(chans["G"], np.float32)
    Y = np.asarray(chans["B"], np.float32)
    Z = np.asarray(chans["A"], np.float32)
    D = np.asarray(chans["R"], np.float32)

    # --- Read PNG for RGB ---
    rgb_img = imageio.imread(png_path)[:, :, :3]  # HxWx3 uint8
    if rgb_img.shape[0] != X.shape[0] or rgb_img.shape[1] != X.shape[1]:
        raise ValueError(f"Resolution mismatch: EXR {X.shape} vs PNG {rgb_img.shape}")

    # Validity + depth window
    mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & np.isfinite(D)
    if depth_min is not None:
        mask &= D >= float(depth_min)
    if depth_max is not None:
        mask &= D <= float(depth_max)

    # Subsample
    stride = max(1, int(stride))
    if stride > 1:
        rr = np.zeros_like(mask, dtype=bool)
        rr[::stride, ::stride] = True
        mask &= rr

    # Flatten arrays
    x = X[mask].ravel()
    y = Y[mask].ravel()
    z = Z[mask].ravel()
    d = D[mask].ravel()
    colors = rgb_img[mask].reshape(-1, 3)

    if negate_z:
        z = -z

    # Optional random cap
    if isinstance(limit, int) and x.size > limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(x.size, size=int(limit), replace=False)
        x, y, z, colors = x[idx], y[idx], z[idx], colors[idx]

    print(f"[INFO] Plotting {x.size:,} points from {exr_path.name} + {png_path.name}")

    if show_plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, s=1, c=colors / 255.0)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Equal aspect
        rng = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
        mid = np.array([x.mean(), y.mean(), z.mean()])
        rad = 0.5 * rng.max()
        ax.set_xlim(mid[0] - rad, mid[0] + rad)
        ax.set_ylim(mid[1] - rad, mid[1] + rad)
        ax.set_zlim(mid[2] - rad, mid[2] + rad)
        ax.set_box_aspect([1, 1, 1])

        plt.title(f"{exr_path.name} with {png_path.name}")
        plt.tight_layout()
        plt.show()
    else:
        print("[INFO] Skipping interactive 3D plot display.")

    # Determine output PLY path (per-selection if provided)
    out_path = Path(ply_out) if ply_out is not None else Path(PLYRGB_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(x, y, z, out_path, colors)

def save_ply(x, y, z, filename="output.ply", colors=None):
    n_points = x.shape[0]
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for xi, yi, zi in zip(x, y, z):
                f.write(f"{xi} {yi} {zi}\n")
        else:
            for xi, yi, zi, ci in zip(x, y, z, colors):
                r, g, b = ci
                f.write(f"{xi} {yi} {zi} {r} {g} {b}\n")
    print(f"[INFO] Saved {n_points:,} points to {filename}")

def main(show_plot: bool = True):
    if SELECTED_ITEMS:
        for sel in SELECTED_ITEMS:
            print(f"\n[INFO] Processing selection: slice={sel.slice} frame={sel.frame} eye={sel.eye}")
            plot_xyz_with_passthrough(
                exr_path=Path(sel.depth_path),
                png_path=Path(sel.rgb_path),
                stride=STRIDE,
                depth_min=DEPTH_MIN,
                depth_max=DEPTH_MAX,
                negate_z=NEGATE_Z,
                limit=LIMIT,
                ply_out=sel.ply_path,
                show_plot=show_plot,
            )
    else:
        plot_xyz_with_passthrough(
            exr_path=Path(DEPTH_PATH),
            png_path=Path(RGB_PATH),
            stride=STRIDE,
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
            negate_z=NEGATE_Z,
            limit=LIMIT,
            ply_out=PLYRGB_PATH,
            show_plot=show_plot,
        )

if __name__ == "__main__":
    main()
