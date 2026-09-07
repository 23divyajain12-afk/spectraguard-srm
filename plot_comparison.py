from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("data/outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bic = np.load(OUTPUT_DIR / "bicubic" / "bicubic.npz")
fused = np.load(OUTPUT_DIR / "fused" / "fused.npz")

bands = list(fused["band_names"])
# --- True Color Natural RGB Bands ---
r, g, b = bands.index("B04"), bands.index("B03"), bands.index("B02")

def get_rgb(data):
    rgb = data[[r, g, b]].transpose(1, 2, 0).astype(np.float32)
    valid = rgb[rgb > 0]
    p2, p98 = np.percentile(valid, (2, 98))
    return np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

rgb_bic = get_rgb(bic["data"])
rgb_fused = get_rgb(fused["data"])

# Crop into top-right ground region
H, W, _ = rgb_fused.shape
crop_y1, crop_y2 = int(H * 0.05), int(H * 0.25)
crop_x1, crop_x2 = int(W * 0.75), int(W * 0.95)

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

axs[0].imshow(rgb_bic[crop_y1:crop_y2, crop_x1:crop_x2])
axs[0].set_title("True Color (RGB): Bicubic Baseline (10m)", fontsize=13)
axs[0].axis("off")

axs[1].imshow(rgb_fused[crop_y1:crop_y2, crop_x1:crop_x2])
axs[1].set_title("True Color (RGB): SpectraGuard-SRM (2.5m)", fontsize=13)
axs[1].axis("off")

plt.tight_layout()
output_file = str(OUTPUT_DIR / "ground_detail_true_color.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved true color comparison to: {output_file}")