# exr_channel_analyzer.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import DEPTHABS_PATH
import exr_utils as eu  

# Preferred display order & simple styling
PREFERRED = ("R", "G", "B", "A")
CMAP_FOR = {"R": "inferno", "G": "inferno", "B": "gray", "A": "gray"}
TITLE_FOR = {"R": "Red (depth)", "G": "Green", "B": "Blue", "A": "Alpha"}

def analyze_and_plot_exr(filepath: str, preferred=PREFERRED):
    """
    Use exr_utils to list and read channels from an EXR file,
    then visualize each channel with a colorbar.
    """
    print(f"--- Analyzing {filepath} ---")
    p = Path(filepath)
    if not p.exists():
        print(f"Error: File not found at '{filepath}'")
        return

    # Discover channels present in the file
    available = eu.list_channels(str(p))
    if not available:
        print("No channels found in EXR.")
        return

    # Keep requested order, then append any extras the file may have
    order = [c for c in preferred if c in available] + [c for c in available if c not in preferred]

    # Read channels via exr_utils (promotes HALF->float32 automatically)
    imgs = {c: eu.read_channel(str(p), c) for c in order}

    # Build subplot grid (2 columns unless only 1 channel)
    n = len(order)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, order):
        im = ax.imshow(imgs[c], cmap=CMAP_FOR.get(c, "gray"))
        ax.set_title(TITLE_FOR.get(c, c), fontsize=12)
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        ax.set_axis_off()

    # Hide any unused axes if grid > channel count
    for ax in axes[len(order):]:
        ax.set_visible(False)

    fig.suptitle(f"Channel Analysis: {p.name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print(f"--- Finished analyzing {filepath} ---\n")


if __name__ == "__main__":
    analyze_and_plot_exr(DEPTHABS_PATH)
