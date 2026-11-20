#!/usr/bin/env python3
# Copyright (c) 2026 Anthony Akens (kraegar)
# SPDX-License-Identifier: MIT
"""
evaluate.py — Rank stacked full-disk solar TIFFs (e.g. from AutoStakkert!4)
by Sobel sharpness, Immerkær noise/SNR, on-band etalon contrast, and wind isotropy.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import scipy.ndimage as ndi
import skimage.filters as skfilters
import tifffile


def evaluate_session_capture(filepath: Path) -> dict:
    img = tifffile.imread(filepath).astype(np.float64)
    if img.ndim == 3:
        if img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    h, w = img.shape
    img_norm = img / (np.max(img) + 1e-8)

    # Otsu threshold cleanly separates solar disk from sky regardless of ROI fill factor
    thresh = skfilters.threshold_otsu(img_norm)
    disk_mask = img_norm > thresh

    # Erode by 3% of the smaller image dimension to stay strictly inside the limb
    erode_px = max(25, int(min(h, w) * 0.03))
    interior = ndi.distance_transform_edt(disk_mask) > erode_px

    if not np.any(interior):
        raise ValueError(f"Could not detect solar disk interior in {filepath.name}")

    # 1. FOCUS METRIC: High-frequency spatial gradient (Sobel) + wind isotropy
    gx = np.abs(ndi.sobel(img_norm, axis=1))[interior]
    gy = np.abs(ndi.sobel(img_norm, axis=0))[interior]

    mean_gx = float(np.mean(gx))
    mean_gy = float(np.mean(gy))
    focus_sharpness = float(np.sqrt(mean_gx**2 + mean_gy**2) * 1000.0)
    isotropy = min(mean_gx, mean_gy) / (max(mean_gx, mean_gy) + 1e-8)

    # 2. ETALON CONTRAST: On-band filament/plage contrast strictly inside the disk
    blur_sigma = max(h, w) / 15.0
    broad_illumination = ndi.gaussian_filter(img_norm, sigma=blur_sigma)
    flattened_disk = (img_norm / (broad_illumination + 1e-6))[interior]

    f_p5, f_p95 = np.percentile(flattened_disk, [5, 95])
    etalon_contrast = float((f_p95 - f_p5) * 100.0)

    # 3. NOISE FLOOR & SNR (Immerkær fast noise variance kernel)
    noise_kernel = np.array(
        [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]],
        dtype=np.float64,
    )
    lap_noise = ndi.convolve(img_norm, noise_kernel, mode="reflect")
    sigma_noise = float(np.sqrt(np.mean(lap_noise[interior] ** 2) / 36.0) * 1000.0)
    focus_snr = focus_sharpness / (sigma_noise + 1e-6)

    # 4. COMPOSITE SCORE
    score = focus_snr * (etalon_contrast / 10.0) * (isotropy ** 2)

    return {
        "file": str(filepath.name),
        "pixels": w * h,
        "res": f"{w}x{h}",
        "focus": focus_sharpness,
        "noise": sigma_noise,
        "snr": focus_snr,
        "etalon_contr": etalon_contrast,
        "wind_iso": isotropy,
        "score": score,
    }


def collect_tif_files(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in {".tif", ".tiff"} else []
    if not target.is_dir():
        raise FileNotFoundError(f"Path does not exist: {target}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in target.glob(pattern)
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rank stacked solar TIFFs by focuser sharpness, SNR, etalon contrast, and wind isotropy."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Directory containing stacked .tif/.tiff files (or path to a single TIFF file).",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    args = parser.parse_args()

    try:
        files = collect_tif_files(args.path, args.recursive)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"No .tif/.tiff files found in '{args.path}'.", file=sys.stderr)
        sys.exit(1)

    results = []
    for f in files:
        try:
            results.append(evaluate_session_capture(f))
        except Exception as e:
            print(f"Skipping {f.name}: {e}", file=sys.stderr)

    if not results:
        print("No valid solar TIFFs could be evaluated.", file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda x: (x["pixels"], x["score"]), reverse=True)

    print(
        f"{'Rank':<5} {'File':<32} {'Res':<10} {'Focus':>7} {'Noise':>7} {'SNR':>7} "
        f"{'EtalonC':>8} {'Iso':>6} {'Score':>8}"
    )
    print("-" * 100)
    for idx, r in enumerate(results, 1):
        print(
            f"{idx:<5} {r['file']:<32} {r['res']:<10} {r['focus']:>7.3f} {r['noise']:>7.4f} "
            f"{r['snr']:>7.2f} {r['etalon_contr']:>8.2f} {r['wind_iso']:>6.3f} {r['score']:>8.2f}"
        )


if __name__ == "__main__":
    main()
