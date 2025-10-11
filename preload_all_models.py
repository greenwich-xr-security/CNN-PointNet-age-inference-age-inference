"""Preload and cache all EfficientNet variants used by `train_age.py`.

Running this script downloads the ImageNet-pretrained weights for every
supported EfficientNet backbone into the project-local cache so future
training runs can start without reaching out to the network.
"""

from __future__ import annotations

import torch
from torchvision import models

from train_age import (
    CHECKPOINTS_DIR,
    EFFICIENTNET_IMG_SIZES,
    ensure_local_torch_home,
    get_default_efficientnet_weights,
    get_local_weights_path,
)


def _download_variant(variant: str) -> None:
    weights = get_default_efficientnet_weights(variant)
    local_path = get_local_weights_path(variant)
    if local_path is not None and local_path.exists():
        print(f"EfficientNet-{variant.upper()} already cached at {local_path}.")
        return

    print(f"Downloading EfficientNet-{variant.upper()} weights...")
    if weights is None:
        # Fallback for older torchvision releases: instantiate once to trigger caching.
        backbone_builder = getattr(models, f"efficientnet_{variant}")
        backbone_builder(pretrained=True)
        print(
            "  Used fallback download path. If training still reports missing weights, "
            "upgrade torchvision or download manually."
        )
        return

    state_dict = weights.get_state_dict(progress=True, check_hash=True)
    if local_path is None:
        # If torchvision does not expose the filename, save a copy for deterministic use.
        fallback_path = CHECKPOINTS_DIR / f"efficientnet_{variant}.pth"
        torch.save(state_dict, fallback_path)
        print(f"  Saved fallback copy to {fallback_path}.")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        torch.save(state_dict, local_path)
    print(f"  Cached at {local_path}.")


def preload_all_models() -> None:
    ensure_local_torch_home()
    for variant in sorted(EFFICIENTNET_IMG_SIZES.keys()):
        _download_variant(variant)
    print("All EfficientNet weights are cached locally.")


if __name__ == "__main__":
    preload_all_models()
