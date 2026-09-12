from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """Extracts True Color RGB [B04, B03, B02] with 1st-99th percentile stretch."""
    rgb = arr[:3].transpose(1, 2, 0).astype(np.float32)
    rgb = rgb[:, :, [2, 1, 0]]
    p1, p99 = np.percentile(rgb, (1, 99))
    if p99 > p1:
        return np.clip((rgb - p1) / (p99 - p1), 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)

def main():
    out_dir = Path("data/outputs")
    orig_path = out_dir / "validation" / "original.npz"
    fused_path = out_dir / "fused" / "fused.npz"

    if not orig_path.exists() or not fused_path.exists():
        print("Missing required .npz files in data/outputs")
        return

    # Load raw arrays
    raw_orig = np.load(orig_path)["data"]   # Shape: (5, 512, 512)
    raw_fused = np.load(fused_path)["data"] # Shape: (5, 2048, 2048)

    rgb_orig = normalize_rgb(raw_orig)      # (512, 512, 3)
    rgb_fused = normalize_rgb(raw_fused)    # (2048, 2048, 3)

    # -------------------------------------------------------------------------
    # Option 1: True Spatial Resolution Comparison (1:1 Ground Coverage Scale)
    # The figure allocates space matching the exact 1:4 pixel dimension ratio.
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(18, 12), 
        dpi=300, 
        gridspec_kw={"width_ratios": [1, 4]}
    )
    plt.subplots_adjust(wspace=0.08, left=0.03, right=0.97, top=0.90, bottom=0.05)

    # Left: Native 512x512
    axes[0].imshow(rgb_orig, interpolation="nearest")
    axes[0].set_title(
        f"Original Sentinel-2\n{rgb_orig.shape[1]}×{rgb_orig.shape[0]} (10m GSD)",
        fontsize=13,
        fontweight="bold",
        pad=10
    )
    axes[0].axis("off")

    # Right: 2048x2048 Super-Resolved (4x width/height)
    axes[1].imshow(rgb_fused, interpolation="lanczos")
    axes[1].set_title(
        f"SpectraGuard-SRM Fused\n{rgb_fused.shape[1]}×{rgb_fused.shape[0]} (2.5m GSD)",
        fontsize=15,
        fontweight="bold",
        pad=10
    )
    axes[1].axis("off")

    prop_path = out_dir / "proportional_512_vs_2048.png"
    plt.savefig(prop_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved proportional scale view: {prop_path}")

    # -------------------------------------------------------------------------
    # Option 2: Equal-Window Viewing with Native Sensor Pixelation
    # Both axes share equal display size, but the 512x512 input retains its
    # coarse 10m detector pixels without any synthetic blurring or interpolation.
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    plt.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.92, bottom=0.04)

    axes[0].imshow(rgb_orig, interpolation="nearest")
    axes[0].set_title(
        f"Original Sentinel-2 Input (10m)\nArray: {rgb_orig.shape[1]}×{rgb_orig.shape[0]}",
        fontsize=15,
        fontweight="bold",
        pad=10
    )
    axes[0].axis("off")

    axes[1].imshow(rgb_fused, interpolation="lanczos")
    axes[1].set_title(
        f"SpectraGuard Super-Resolution (2.5m)\nArray: {rgb_fused.shape[1]}×{rgb_fused.shape[0]}",
        fontsize=15,
        fontweight="bold",
        pad=10
    )
    axes[1].axis("off")

    equal_path = out_dir / "equal_view_512_vs_2048.png"
    plt.savefig(equal_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved equal-window view: {equal_path}")

if __name__ == "__main__":
    main()