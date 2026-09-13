"""
ct_hu_analysis.py
==================
CT Hounsfield Unit (HU) Analysis — single-file version.

What this script does:
1. Converts raw CT pixel data to Hounsfield Units (HU = pixel * slope + intercept)
2. Applies clinical windowing (bone / lung / brain / soft tissue views)
3. Classifies each pixel by tissue type using standard HU reference ranges
4. Generates a synthetic CT phantom (so the script runs with NO real patient
   data needed — no privacy/ethics concerns for a public GitHub repo)
5. Plots and saves: window comparison grid, HU histogram, tissue map

Usage
-----
    python ct_hu_analysis.py                      # runs on built-in synthetic phantom
    python ct_hu_analysis.py --dicom <folder>      # runs on a real DICOM series instead

Requires: numpy, matplotlib   (pydicom only if using --dicom)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# 1. HU conversion
# ---------------------------------------------------------------------------

def pixels_to_hu(pixel_array, slope=1.0, intercept=-1024.0):
    """HU = pixel_value * RescaleSlope + RescaleIntercept"""
    return pixel_array.astype(np.float32) * slope + intercept


def dicom_to_hu(dataset):
    """Convert a pydicom Dataset to an HU array using its own header tags."""
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", -1024.0))
    return pixels_to_hu(dataset.pixel_array, slope, intercept)


def load_dicom_series(folder_path):
    """Load every .dcm file in a folder, sorted by InstanceNumber."""
    import pydicom
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {folder_path}")
    datasets = [pydicom.dcmread(os.path.join(folder_path, f)) for f in files]
    datasets.sort(key=lambda ds: getattr(ds, "InstanceNumber", 0))
    return [dicom_to_hu(ds) for ds in datasets]


# ---------------------------------------------------------------------------
# 2. Windowing (clinical window/level presets)
# ---------------------------------------------------------------------------

WINDOW_PRESETS = {
    "brain": (40, 80),
    "bone": (400, 1800),
    "soft_tissue": (50, 400),
    "lung": (-600, 1500),
}


def apply_window(hu_array, center, width):
    lower, upper = center - width / 2, center + width / 2
    clipped = np.clip(hu_array, lower, upper)
    return ((clipped - lower) / (upper - lower) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Tissue classification by HU range
# ---------------------------------------------------------------------------

HU_RANGES = {
    "air": (-1000, -900),
    "lung": (-900, -500),
    "fat": (-120, -90),
    "water": (-10, 10),
    "soft_tissue": (10, 40),
    "blood_clot": (50, 75),
    "trabecular_bone": (300, 400),
    "cortical_bone": (400, 3000),
}


def classify_tissue(hu_array):
    labels = np.zeros(hu_array.shape, dtype=np.int16)
    for i, (name, (low, high)) in enumerate(HU_RANGES.items(), start=1):
        labels[(hu_array >= low) & (hu_array < high)] = i
    return labels


def tissue_statistics(hu_array):
    total = hu_array.size
    stats = {}
    for name, (low, high) in HU_RANGES.items():
        count = int(np.sum((hu_array >= low) & (hu_array < high)))
        stats[name] = round(100 * count / total, 2)
    stats["unclassified"] = round(max(100 - sum(stats.values()), 0.0), 2)
    return stats


# ---------------------------------------------------------------------------
# 4. Synthetic CT phantom (no real patient data needed)
# ---------------------------------------------------------------------------

def _disk_mask(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2


def generate_ct_phantom(size=512, seed=42):
    """Builds a simplified abdominal-CT-like slice: body, fat ring, spine,
    a lung pocket, a lesion, and a fluid-filled structure, plus scanner noise."""
    rng = np.random.default_rng(seed)
    hu = np.full((size, size), -1000.0, dtype=np.float32)
    center = (size // 2, size // 2)
    body_radius = int(size * 0.42)

    body_mask = _disk_mask(hu.shape, center, body_radius)
    hu[body_mask] = 35.0

    fat_inner = _disk_mask(hu.shape, center, int(body_radius * 0.88))
    hu[body_mask & ~fat_inner] = -100.0

    spine_center = (center[0], int(center[1] + body_radius * 0.45))
    hu[_disk_mask(hu.shape, spine_center, int(size * 0.045))] = 700.0

    lung_center = (int(center[0] - body_radius * 0.35), int(center[1] - body_radius * 0.25))
    hu[_disk_mask(hu.shape, lung_center, int(size * 0.12)) & body_mask] = -800.0

    lesion_center = (int(center[0] + body_radius * 0.3), int(center[1] - body_radius * 0.3))
    hu[_disk_mask(hu.shape, lesion_center, int(size * 0.05)) & body_mask] = 65.0

    fluid_center = (int(center[0] - body_radius * 0.15), int(center[1] + body_radius * 0.1))
    hu[_disk_mask(hu.shape, fluid_center, int(size * 0.04)) & body_mask] = 5.0

    hu += rng.normal(0, 12, hu.shape).astype(np.float32)
    return hu


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def plot_window_grid(hu_array, save_path):
    fig, axes = plt.subplots(1, len(WINDOW_PRESETS), figsize=(4 * len(WINDOW_PRESETS), 4))
    for ax, (name, (center, width)) in zip(axes, WINDOW_PRESETS.items()):
        ax.imshow(apply_window(hu_array, center, width), cmap="gray")
        ax.set_title(f"{name}\n(C={center}, W={width})")
        ax.axis("off")
    fig.suptitle("Same CT slice under different clinical windows")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_hu_histogram(hu_array, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(hu_array.ravel(), bins=200, range=(-1050, 1050), color="steelblue")
    ax.set_xlabel("Hounsfield Units (HU)")
    ax.set_ylabel("Pixel count")
    ax.set_title("HU distribution with tissue reference ranges")
    colors = plt.cm.tab10(np.linspace(0, 1, len(HU_RANGES)))
    handles = []
    for (name, (low, high)), color in zip(HU_RANGES.items(), colors):
        ax.axvspan(low, high, color=color, alpha=0.15)
        handles.append(Patch(facecolor=color, alpha=0.4, label=f"{name} ({low} to {high})"))
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_tissue_map(hu_array, save_path):
    labels = classify_tissue(hu_array)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(apply_window(hu_array, 50, 400), cmap="gray")
    axes[0].set_title("Soft-tissue window")
    axes[0].axis("off")
    im = axes[1].imshow(labels, cmap="tab10", vmin=0, vmax=len(HU_RANGES))
    axes[1].set_title("Tissue classification map")
    axes[1].axis("off")
    names = ["unclassified"] + list(HU_RANGES.keys())
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(len(names)), fraction=0.046)
    cbar.ax.set_yticklabels(names, fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CT Hounsfield Unit analysis")
    parser.add_argument("--dicom", type=str, default=None, help="Path to a folder of .dcm files")
    parser.add_argument("--out", type=str, default="output", help="Output folder for plots")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dicom:
        print(f"Loading DICOM series from {args.dicom} ...")
        slices = load_dicom_series(args.dicom)
        hu = slices[len(slices) // 2]
    else:
        print("No --dicom folder given, using the built-in synthetic phantom.")
        hu = generate_ct_phantom()

    print(f"HU range in this slice: {hu.min():.1f} to {hu.max():.1f}")

    plot_window_grid(hu, os.path.join(args.out, "windows.png"))
    plot_hu_histogram(hu, os.path.join(args.out, "histogram.png"))
    plot_tissue_map(hu, os.path.join(args.out, "tissue_map.png"))

    print("\nTissue composition (% of slice):")
    for name, pct in sorted(tissue_statistics(hu).items(), key=lambda kv: -kv[1]):
        if pct > 0:
            print(f"  {name:<18} {pct:>6.2f}%")

    print(f"\nDone. Plots saved to ./{args.out}/")


if __name__ == "__main__":
    main()
