#!/usr/bin/env python3
# Copyright (c) 2026 Anthony Akens (kraegar)
# SPDX-License-Identifier: MIT

"""
solar_auto.py — Automated Reference-Grade Solar Image Processor
Matches top AstroBin / SolarChat workflows:
  1. Sub-pixel 2nd-order harmonic RANSAC limb lock
  2. Seeing-adaptive unclipped Lucy-Richardson deconvolution
  3. Non-destructive off-limb prominence extraction (inward-ray-clamped wavelets)
  4. Etalon sweet-spot radial + 2D flat-fielding & direct-polar Harmonic FNRGF
  5. 3-zone auto-detected dynamic range stretch with multiple polarity styles
  6. Per-channel incandescent CIELAB RGB curve solver
"""

__version__ = "1.0.0"

import argparse
import math
import os
from pathlib import Path
import time

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import astropy.io.fits as fits
import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize
import skimage.color as skcolor
import skimage.filters as skfilters
import skimage.measure as skmeasure
import skimage.restoration as skrest
import skimage.transform as sktrans
import tifffile


# ==============================================================================
# 1. I/O
# ==============================================================================

def load_solar_image(filepath: Path) -> tuple[np.ndarray, dict]:
    ext = filepath.suffix.lower()
    meta = {}

    if ext in {".fit", ".fits", ".fz"}:
        with fits.open(filepath) as hdul:
            data = hdul[0].data if hdul[0].data is not None else hdul[1].data
            meta = dict(hdul[0].header)
        img = np.nan_to_num(data.astype(np.float32))
    elif ext in {".tif", ".tiff"}:
        img = tifffile.imread(filepath).astype(np.float32)
    else:
        img = iio.imread(filepath).astype(np.float32)

    if img.ndim == 3:
        if img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    max_val = float(np.max(img))
    if max_val > 255.0:
        img /= 65535.0 if max_val > 4095.0 else 4095.0
    elif max_val > 1.0:
        img /= 255.0

    return np.clip(img, 0.0, 1.0).astype(np.float32), meta


def _safe_write(write_fn, target_path: Path, *args, **kwargs):
    """Writes via an atomic temp file with retry so Windows viewer locks never raise Errno 22."""
    tmp_path = target_path.with_name(f".tmp_{target_path.name}")
    write_fn(tmp_path, *args, **kwargs)
    for attempt in range(5):
        try:
            os.replace(tmp_path, target_path)
            return
        except OSError:
            time.sleep(0.15 * (attempt + 1))
    # Final fallback: direct write
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass
    write_fn(target_path, *args, **kwargs)


def save_outputs(
    out_dir: Path, stem: str, gray: np.ndarray, color_rgb: np.ndarray, cfg: dict, style_suffix: str = ""
):
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = cfg["pipeline"]["output_formats"]

    if cfg["pipeline"]["save_grayscale"]:
        if "tiff16" in formats:
            p = out_dir / f"{stem}_mono{style_suffix}.tif"
            _safe_write(
                tifffile.imwrite,
                p,
                (np.clip(gray, 0, 1) * 65535).astype(np.uint16),
                compression="zlib",
            )
            print(f"  -> Wrote: {p}")
        if "png8" in formats:
            p = out_dir / f"{stem}_mono{style_suffix}.png"
            _safe_write(
                iio.imwrite,
                p,
                (np.clip(gray, 0, 1) * 255).astype(np.uint8),
            )
            print(f"  -> Wrote: {p}")

    if cfg["pipeline"]["save_color"]:
        if "tiff16" in formats:
            p = out_dir / f"{stem}_color{style_suffix}.tif"
            _safe_write(
                tifffile.imwrite,
                p,
                (np.clip(color_rgb, 0, 1) * 65535).astype(np.uint16),
                photometric="rgb",
                compression="zlib",
            )
            print(f"  -> Wrote: {p}")
        if "png8" in formats:
            p = out_dir / f"{stem}_color{style_suffix}.png"
            _safe_write(
                iio.imwrite,
                p,
                (np.clip(color_rgb, 0, 1) * 255).astype(np.uint8),
            )
            print(f"  -> Wrote: {p}")


# ==============================================================================
# 2. SUB-PIXEL LIMB DETECTION & SEEING PSF MEASUREMENT
# ==============================================================================

def compute_relative_radius(
    shape: tuple[int, int], cx: float, cy: float, r: float, harmonics: np.ndarray | None = None
) -> np.ndarray:
    """
    Returns normalized radial coordinate r_rel = rr / r_local(theta), where r_local(theta)
    accounts for 2nd-order harmonic ellipticity (differential refraction / optical tilt)
    so the limb lies at r_rel == 1.000 within 0.3 px at all clock angles.
    """
    yy, xx = np.indices(shape, dtype=np.float32)
    dx = xx - cx
    dy = yy - cy
    rr = np.sqrt(dx * dx + dy * dy)
    if harmonics is not None and len(harmonics) == 4:
        c1, s1, c2, s2 = harmonics
        theta = np.arctan2(dy, dx)
        r_local = r + c1 * np.cos(theta) + s1 * np.sin(theta) + c2 * np.cos(2.0 * theta) + s2 * np.sin(2.0 * theta)
        return rr / np.maximum(r_local, 1.0)
    return rr / max(r, 1.0)


def find_subpixel_limb(img: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    h, w = img.shape
    scale = max(1, min(h, w) // 800)
    small = sktrans.rescale(img, 1.0 / scale, anti_aliasing=True) if scale > 1 else img.copy()

    smooth = ndi.gaussian_filter(small, sigma=3.0)
    log_img = np.log1p(smooth * 100.0)
    thresh = skfilters.threshold_otsu(log_img)
    mask = log_img > thresh

    eroded = ndi.binary_erosion(mask)
    edge_y, edge_x = np.nonzero(mask ^ eroded)
    points = np.column_stack([edge_x, edge_y]).astype(np.float64)

    if len(points) > 2500:
        points = points[np.linspace(0, len(points) - 1, 2500, dtype=int)]

    rng = np.random.default_rng(42)
    model, _ = skmeasure.ransac(
        points,
        skmeasure.CircleModel,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=500,
        rng=rng,
    )
    cx, cy, r = [p * scale for p in model.params]

    yy, xx = np.indices(img.shape, dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Measure a FIXED, deterministic limb threshold at 45% of outer-disk intensity (the true optical limb!)
    # and iterate the 2nd-order harmonic contour fit twice so r_rel == 1.000 sits dead-on the physical limb!
    disk_edge_ref = float(np.percentile(img[(rr >= r * 0.95) & (rr < r * 0.97)], 25))
    sky_mask_ref = rr > r * 1.15
    if np.count_nonzero(sky_mask_ref) < 20:
        sky_mask_ref = rr > r * 1.05
    sky_ref = float(np.median(img[sky_mask_ref])) if np.any(sky_mask_ref) else float(np.percentile(img, 2))
    limb_half_val = sky_ref + 0.45 * (disk_edge_ref - sky_ref)

    r_refined = float(r)
    harmonics = np.zeros(4, dtype=np.float32)
    for _ in range(2):
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring_mask = (rr >= r_refined - 12.0) & (rr <= r_refined + 12.0)
        rr_ring = rr[ring_mask]
        img_ring = img[ring_mask]
        theta_ring = np.degrees(np.arctan2(yy[ring_mask] - cy, xx[ring_mask] - cx)) % 360.0

        angles, r_locs = [], []
        r_search = np.arange(r_refined - 10.0, r_refined + 10.0, 0.5)
        for a in range(0, 360, 5):
            m_ang = (theta_ring >= a) & (theta_ring < a + 5)
            if np.count_nonzero(m_ang) < 50:
                continue
            rr_sec = rr_ring[m_ang]
            img_sec = img_ring[m_ang]
            prof = np.array([
                np.percentile(img_sec[(rr_sec >= rb - 0.35) & (rr_sec < rb + 0.35)], 25)
                if np.any((rr_sec >= rb - 0.35) & (rr_sec < rb + 0.35)) else limb_half_val
                for rb in r_search
            ])
            idx = np.where(prof < limb_half_val)[0]
            if len(idx) > 0 and idx[0] > 0:
                i0 = int(idx[0] - 1)
                frac = (limb_half_val - prof[i0]) / (prof[i0 + 1] - prof[i0] - 1e-8)
                r_cross = float(r_search[i0] + np.clip(frac, 0.0, 1.0) * 0.5)
                angles.append(np.radians(a + 2.5))
                r_locs.append(r_cross)

        if len(angles) >= 24:
            ang_arr = np.array(angles)
            rloc_arr = np.array(r_locs)
            A = np.column_stack([
                np.ones_like(ang_arr),
                np.cos(ang_arr),
                np.sin(ang_arr),
                np.cos(2.0 * ang_arr),
                np.sin(2.0 * ang_arr),
            ])
            coeffs = np.linalg.lstsq(A, rloc_arr, rcond=None)[0]
            res = rloc_arr - A @ coeffs
            inliers = np.abs(res) < 1.5
            if np.count_nonzero(inliers) >= 18:
                coeffs = np.linalg.lstsq(A[inliers], rloc_arr[inliers], rcond=None)[0]
            cx += float(coeffs[1])
            cy += float(coeffs[2])
            r_refined = float(coeffs[0])
            harmonics = np.array([0.0, 0.0, coeffs[3], coeffs[4]], dtype=np.float32)

    return float(cx), float(cy), r_refined, harmonics


def estimate_seeing_psf(img: np.ndarray, cx: float, cy: float, r: float) -> float:
    res_s = float(np.clip(r / 680.0, 0.75, 2.0))
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    half_span = 12.0 * res_s
    radii = np.linspace(r - half_span, r + half_span, 97)
    profile = np.array([
        np.percentile(img[(rr >= rad - 0.5) & (rr < rad + 0.5)], 35)
        for rad in radii
    ])
    p_low, p_high = np.min(profile), np.max(profile)
    if p_high <= p_low:
        return float(1.0 * res_s)
    norm_prof = (profile - p_low) / (p_high - p_low)
    idx_90 = np.argmax(norm_prof < 0.90)
    idx_10 = np.argmax(norm_prof < 0.10)
    sigma_raw = abs(radii[idx_10] - radii[idx_90]) / 2.563
    # Subtract the intrinsic chromospheric spicule-skirt width (~1.15 px * res_s) in quadrature
    # so L-R deconvolution targets true optical seeing without over-sharpening 1x mottles.
    sigma_limb_intrinsic = 1.15 * res_s
    sigma = math.sqrt(max(sigma_raw ** 2 - sigma_limb_intrinsic ** 2, (0.85 * res_s) ** 2))
    return float(np.clip(sigma, 0.85 * res_s, 1.45 * res_s))


# ==============================================================================
# 3. SKY CALIBRATION, FLAT-FIELDING, CNRGF & MULTI-SCALE SHARPENING
# ==============================================================================

def subtract_local_offlimb_scatter(
    img: np.ndarray, r_rel: np.ndarray, cx: float, cy: float, strength: float = 1.0
) -> np.ndarray:
    """
    Vectorized polar off-limb scatter subtraction:
    Measures and subtracts smooth angular atmospheric/instrument scatter haze outside the
    chromospheric limb (r_rel >= 1.003) using a 70-deg morphological opening filter so narrow
    prominences and fine tendrils remain intact while asymmetric haze (e.g. 3-5 o'clock) is removed.
    """
    if strength <= 0.0:
        return img

    yy, xx = np.indices(img.shape, dtype=np.float32)
    theta_deg = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0

    r_knots = np.array([1.005, 1.010, 1.016, 1.023, 1.032, 1.045, 1.065, 1.095, 1.140, 1.250], dtype=np.float32)
    n_ang = 36
    ang_step = 360.0 / n_ang

    # Pre-slice off-limb annulus once for fast vectorized sector statistics
    ann_mask = (r_rel >= 1.002) & (r_rel <= 1.260)
    r_ann = r_rel[ann_mask]
    t_ann = theta_deg[ann_mask]
    i_ann = img[ann_mask]
    ia_ann = np.clip((t_ann / ang_step).astype(np.int32), 0, n_ang - 1)

    bg_grid = np.zeros((len(r_knots), n_ang), dtype=np.float32)
    for ir, rk in enumerate(r_knots):
        dr = 0.0025 if rk < 1.025 else 0.006
        m_r = (r_ann >= rk - dr) & (r_ann < rk + dr)
        if np.count_nonzero(m_r) < 36:
            continue
        q_pct = 28 if rk < 1.014 else (42 if rk < 1.026 else 58)
        i_r = i_ann[m_r]
        ia_r = ia_ann[m_r]
        fallback_q = float(np.percentile(i_r, q_pct))
        for ia in range(n_ang):
            sec_vals = i_r[ia_r == ia]
            bg_grid[ir, ia] = float(np.percentile(sec_vals, q_pct)) if len(sec_vals) >= 5 else fallback_q

    # Morphological opening (minimum filter of 70 deg = 7 sectors) rejects prominences and wide bridge complexes
    # while preserving the broad quadrant-scale atmospheric scatter band
    bg_grid = ndi.minimum_filter1d(bg_grid, size=7, axis=1, mode="wrap")
    bg_grid = ndi.gaussian_filter1d(bg_grid, sigma=2.5, axis=1, mode="wrap")
    bg_grid = ndi.gaussian_filter1d(bg_grid, sigma=0.7, axis=0, mode="nearest")

    # Enforce monotonic radial decay outward from the limb
    for ir in range(1, len(r_knots)):
        bg_grid[ir] = np.minimum(bg_grid[ir], bg_grid[ir - 1])

    # Subtract asymmetric excess scatter near the limb, plus the symmetric near-limb haze floor
    # beyond 1.010R so frames with high atmospheric sky scatter (e.g. 13_17_32 or Drizzle15 stacks)
    # never form a detached glowing ring at 1.010R..1.030R!
    clean_radial_base = np.percentile(bg_grid, 15, axis=1, keepdims=True)
    bg_asym = np.clip(bg_grid - clean_radial_base, 0.0, None)
    far_weight = np.clip((r_knots[:, None] - 1.006) / 0.010, 0.0, 1.0)
    bg_effective = bg_asym + far_weight * clean_radial_base * 0.98

    bg_ext = np.column_stack([bg_effective, bg_effective[:, 0]])

    out = img.copy()
    mask_off = r_rel > 1.003
    r_vals = np.clip(r_rel[mask_off], r_knots[0], r_knots[-1])
    a_vals = theta_deg[mask_off]

    ir_idx = np.clip(np.searchsorted(r_knots, r_vals) - 1, 0, len(r_knots) - 2)
    fr = (r_vals - r_knots[ir_idx]) / np.maximum(r_knots[ir_idx + 1] - r_knots[ir_idx], 1e-6)

    ia_f = a_vals / ang_step
    ia_idx = np.clip(ia_f.astype(np.int32), 0, n_ang - 1)
    fa = ia_f - ia_idx

    b00 = bg_ext[ir_idx, ia_idx]
    b01 = bg_ext[ir_idx, ia_idx + 1]
    b10 = bg_ext[ir_idx + 1, ia_idx]
    b11 = bg_ext[ir_idx + 1, ia_idx + 1]
    bg_interp = (1.0 - fr) * ((1.0 - fa) * b00 + fa * b01) + fr * ((1.0 - fa) * b10 + fa * b11)

    t_ramp = np.clip((r_vals - 1.003) / 0.007, 0.0, 1.0)
    outer_boost = 1.0 + 0.22 * np.clip((r_vals - 1.010) / 0.015, 0.0, 1.0)
    ramp = t_ramp * t_ramp * (3.0 - 2.0 * t_ramp) * strength * outer_boost
    out[mask_off] = np.clip(out[mask_off] - ramp * bg_interp, 0.0, None)

    local_support = ndi.uniform_filter(out, size=7)
    out = np.where((r_rel > 1.011) & (local_support < 0.0024) & (out < 0.0075), 0.0, out)
    return out


def apply_harmonic_fnrgf(
    img: np.ndarray,
    r_rel: np.ndarray,
    cx: float,
    cy: float,
    min_gain: float = 1.35,
    max_gain: float = 2.45,
    blend: float = 0.65,
) -> np.ndarray:
    """
    True Druckmüller Fourier Normalizing Radial Graded Filter (FNRGF) / SolarFinish CNRGF
    implemented directly on the ellipticity-corrected (r_rel, theta) polar grid:
      - Computes local angular mean mu(r_rel, theta) and standard deviation sigma(r_rel, theta)
        over 40-deg sliding arcs along constant r_rel shells (zero cv.linearPolar interpolation blur!).
      - Adaptive gain G(r_rel, theta) = 1 / norm_sigma equalizes micro-contrast across the entire
        solar disk and out to the chromospheric limb, giving filaments, active regions, and
        prominences effortless 3D pop without harsh high-frequency ringing!
    """
    if blend <= 0.0:
        return img

    yy, xx = np.indices(img.shape, dtype=np.float32)
    theta_deg = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0

    n_r = 160
    n_a = 72  # 5-deg angular bins
    r_max = 1.06
    r_knots = np.linspace(0.0, r_max, n_r, dtype=np.float32)
    ang_step = 360.0 / n_a

    mask_valid = r_rel <= r_max
    r_v = r_rel[mask_valid]
    t_v = theta_deg[mask_valid]
    i_v = img[mask_valid]

    ir_v = np.clip((r_v / r_max * (n_r - 1)).astype(np.int32), 0, n_r - 1)
    ia_v = np.clip((t_v / ang_step).astype(np.int32), 0, n_a - 1)
    flat_idx = ir_v * n_a + ia_v

    counts = np.bincount(flat_idx, minlength=n_r * n_a).astype(np.float32)
    sums = np.bincount(flat_idx, weights=i_v, minlength=n_r * n_a).astype(np.float32)
    sums2 = np.bincount(flat_idx, weights=i_v * i_v, minlength=n_r * n_a).astype(np.float32)

    counts_2d = counts.reshape(n_r, n_a)
    mean_grid = sums.reshape(n_r, n_a) / np.maximum(counts_2d, 1.0)
    var_grid = np.maximum(sums2.reshape(n_r, n_a) / np.maximum(counts_2d, 1.0) - mean_grid ** 2, 0.0)

    # Fill any empty inner bins near r_rel -> 0 with the radial ring average
    empty = counts_2d < 1.0
    if np.any(empty):
        ring_mean = np.sum(sums.reshape(n_r, n_a), axis=1, keepdims=True) / np.maximum(
            np.sum(counts_2d, axis=1, keepdims=True), 1.0
        )
        mean_grid = np.where(empty, ring_mean, mean_grid)

    # Smooth angularly over ~45-deg arcs (sigma=3.0 on 5-deg bins) with periodic wraparound
    # and slightly radially (sigma=1.0) to form the smooth CNRGF/FNRGF mean & std-dev surfaces
    mu_smooth = ndi.gaussian_filter1d(mean_grid, sigma=3.2, axis=1, mode="wrap")
    mu_smooth = ndi.gaussian_filter1d(mu_smooth, sigma=1.0, axis=0, mode="nearest")

    # Local angular variance (within-bin variance + between-bin variance across the 45-deg arc)
    sq_diff = (mean_grid - mu_smooth) ** 2 + var_grid
    var_smooth = ndi.gaussian_filter1d(sq_diff, sigma=3.2, axis=1, mode="wrap")
    var_smooth = ndi.gaussian_filter1d(var_smooth, sigma=1.2, axis=0, mode="nearest")
    std_grid = np.sqrt(np.maximum(var_smooth, 1e-8))

    # Normalize std_grid inside the disk (0.08R..0.98R) into [1/max_gain, 1/min_gain]
    # exactly as SolarFinish's get_std_dev_scaler(min_recip, max_recip) does!
    disk_std = std_grid[(r_knots >= 0.08) & (r_knots <= 0.98), :]
    s_low = float(np.percentile(disk_std, 5))
    s_high = float(np.percentile(disk_std, 95))
    std_norm01 = np.clip((std_grid - s_low) / max(s_high - s_low, 1e-5), 0.0, 1.0)
    inv_max = 1.0 / max_gain
    inv_min = 1.0 / min_gain
    scaled_std = inv_max + std_norm01 * (inv_min - inv_max)
    gain_grid = 1.0 / scaled_std

    # Taper FNRGF enhancement smoothly to 1.0 before the optical limb cliff (0.980R -> 0.995R)
    # so radial binning never creates a bright overshoot ring at 0.990R!
    limb_taper = 1.0 - np.clip((r_knots[:, None] - 0.980) / 0.015, 0.0, 1.0)
    gain_grid = 1.0 + blend * limb_taper * (gain_grid - 1.0)

    # Also compute the radial ring median so FNRGF/CNRGF pulls azimuthal etalon shading
    # (e.g. off-band dark quadrants near the limb) toward the ring average!
    ring_med = np.median(mu_smooth, axis=1, keepdims=True)
    az_level_taper = 1.0 - np.clip((r_knots[:, None] - 0.978) / 0.014, 0.0, 1.0)
    mu_target = mu_smooth + 0.65 * az_level_taper * (ring_med - mu_smooth)

    mu_ext = np.column_stack([mu_smooth, mu_smooth[:, 0]])
    target_ext = np.column_stack([mu_target, mu_target[:, 0]])
    gain_ext = np.column_stack([gain_grid, gain_grid[:, 0]])

    # Vectorized bilinear evaluation back onto the full-resolution image
    ir_f = np.clip(r_v / r_max * (n_r - 1), 0.0, n_r - 1.0001)
    ir0 = ir_f.astype(np.int32)
    fr = ir_f - ir0

    ia_f = np.clip(t_v / ang_step, 0.0, n_a - 0.0001)
    ia0 = ia_f.astype(np.int32)
    fa = ia_f - ia0

    mu_interp = (1.0 - fr) * ((1.0 - fa) * mu_ext[ir0, ia0] + fa * mu_ext[ir0, ia0 + 1]) + fr * (
        (1.0 - fa) * mu_ext[ir0 + 1, ia0] + fa * mu_ext[ir0 + 1, ia0 + 1]
    )
    target_interp = (1.0 - fr) * ((1.0 - fa) * target_ext[ir0, ia0] + fa * target_ext[ir0, ia0 + 1]) + fr * (
        (1.0 - fa) * target_ext[ir0 + 1, ia0] + fa * target_ext[ir0 + 1, ia0 + 1]
    )
    gain_interp = (1.0 - fr) * ((1.0 - fa) * gain_ext[ir0, ia0] + fa * gain_ext[ir0, ia0 + 1]) + fr * (
        (1.0 - fa) * gain_ext[ir0 + 1, ia0] + fa * gain_ext[ir0 + 1, ia0 + 1]
    )

    # Softly compress only extreme outlier spikes (>0.42) so true dark active-region canopies
    # and bright filaments keep 100% of their natural tail separation for the auto-stretcher!
    diff = i_v - mu_interp
    enhanced_v = target_interp + gain_interp * (0.42 * np.tanh(diff / 0.42))

    out = img.copy()
    out[mask_valid] = np.clip(enhanced_v, 0.0, None)
    return out


def extract_offlimb_prominences(
    deconv: np.ndarray,
    r_rel: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    harmonics: np.ndarray,
    sky_med: float,
    sky_std: float,
) -> np.ndarray:
    """
    Extracts full-detail off-limb prominences and the chromospheric spicule fringe from the
    unclipped deconvolved image:
      1. Measures smooth angular/radial atmospheric sky scatter beyond 1.028R (72-deg min filter)
         and holds it flat inward so the spicule forest and prominence roots are never eaten away.
      2. Uses inward-ray clamping from 1.010R as the source for 4-scale Gaussian wavelets so the
         bright solar disk never bleeds outward to create a dark/bright limb-cliff halo.
      3. Applies a soft noise gate so empty sky stays pure black while every feathery prominence
         hair and internal dark lane is preserved.
    """
    n_theta = 1440
    n_r = max(128, int(round(r * 1.95)))
    r_max = 1.95
    theta_1d = np.linspace(-np.pi, np.pi, n_theta, endpoint=False, dtype=np.float32)
    r_rel_1d = np.linspace(0.0, r_max, n_r, dtype=np.float32)

    c1, s1, c2, s2 = harmonics if (harmonics is not None and len(harmonics) == 4) else (0.0, 0.0, 0.0, 0.0)
    r_loc_1d = (
        r + c1 * np.cos(theta_1d) + s1 * np.sin(theta_1d) + c2 * np.cos(2.0 * theta_1d) + s2 * np.sin(2.0 * theta_1d)
    )[:, None]
    rr_grid = r_loc_1d * r_rel_1d[None, :]
    map_x = (cx + rr_grid * np.cos(theta_1d)[:, None]).astype(np.float32)
    map_y = (cy + rr_grid * np.sin(theta_1d)[:, None]).astype(np.float32)

    polar_raw = ndi.map_coordinates(deconv, [map_y, map_x], order=1, mode="nearest")
    k_bg = (n_theta // 5) | 1
    bg_polar = ndi.minimum_filter1d(polar_raw, size=k_bg, axis=0, mode="wrap")
    bg_polar = ndi.gaussian_filter1d(bg_polar, sigma=k_bg * 0.35, axis=0, mode="wrap")
    bg_polar = ndi.gaussian_filter1d(bg_polar, sigma=2.5, axis=1, mode="nearest")

    ir_sky = min(n_r - 2, max(1, int(round(1.028 / r_max * (n_r - 1)))))
    ir_decay_end = min(n_r - 1, max(ir_sky + 1, int(round(1.18 / r_max * (n_r - 1)))))
    for ir in range(ir_sky + 1, ir_decay_end + 1):
        bg_polar[:, ir] = np.minimum(bg_polar[:, ir], bg_polar[:, ir - 1])
    for ir in range(ir_sky - 1, -1, -1):
        bg_polar[:, ir] = bg_polar[:, ir_sky]
    bg_polar = np.maximum(bg_polar, sky_med)

    yy, xx = np.indices(deconv.shape, dtype=np.float32)
    theta_img = np.arctan2(yy - cy, xx - cx)
    map_ir = np.clip(r_rel / r_max * (n_r - 1), 0.0, n_r - 1).astype(np.float32)
    map_it = (((theta_img + np.pi) / (2.0 * np.pi)) * n_theta) % n_theta
    bg_sky_2d = ndi.map_coordinates(bg_polar, [map_it, map_ir], order=1, mode="wrap")

    base_clean = np.clip(deconv - (bg_sky_2d + 0.95 * sky_std), 0.0, None)

    # Inward-ray clamping from 1.010R for radial-cliff-free 2D prominence wavelets
    polar_clean = np.clip(polar_raw - (bg_polar + 0.95 * sky_std), 0.0, None)
    polar_clamped = polar_clean.copy()
    ir_1010 = min(n_r - 4, max(2, int(round(1.010 / r_max * (n_r - 1)))))
    ir_1005 = min(n_r - 2, max(1, int(round(1.005 / r_max * (n_r - 1)))))
    ref_ceil = ndi.maximum_filter1d(np.max(polar_clean[:, ir_1010 - 1 : ir_1010 + 3], axis=1), size=5, mode="wrap")
    ref_ceil = ndi.gaussian_filter1d(ref_ceil, sigma=1.5, mode="wrap")
    spicule_floor = float(np.percentile(polar_clean[:, ir_1005], 40)) * 0.50
    ref_ceil = np.maximum(ref_ceil, spicule_floor)
    for ir in range(ir_1010 - 1, -1, -1):
        ref_ceil = ref_ceil * 1.018
        polar_clamped[:, ir] = np.minimum(polar_clean[:, ir], ref_ceil)

    clamped_2d = ndi.map_coordinates(polar_clamped, [map_it, map_ir], order=1, mode="wrap")
    wavelet_src = np.where(r_rel >= 1.010, base_clean, clamped_2d)

    res_s = float(np.clip(r / 680.0, 0.75, 2.0))
    pg08 = ndi.gaussian_filter(wavelet_src, sigma=0.75 * res_s)
    pg16 = ndi.gaussian_filter(wavelet_src, sigma=1.60 * res_s)
    pg34 = ndi.gaussian_filter(wavelet_src, sigma=3.40 * res_s)
    pg70 = ndi.gaussian_filter(wavelet_src, sigma=7.00 * res_s)
    detail_2d = 2.6 * (wavelet_src - pg08) + 1.9 * (pg08 - pg16) + 1.1 * (pg16 - pg34) + 0.5 * (pg34 - pg70)

    prom_gate = np.clip((ndi.gaussian_filter(base_clean, sigma=0.9) - 0.85 * sky_std) / max(1.40 * sky_std, 1e-5), 0.0, 1.0)
    prom_gate = np.where(r_rel < 1.003, 1.0, prom_gate)
    prom_gate = prom_gate * (1.0 - np.clip((r_rel - 1.20) / 0.06, 0.0, 1.0))
    # Suppress 2px sensor/deconvolution edge border so padded preview tiles never show a 1px edge line
    prom_gate[:2, :] = 0.0
    prom_gate[-2:, :] = 0.0
    prom_gate[:, :2] = 0.0
    prom_gate[:, -2:] = 0.0
    prom_detailed = np.maximum(base_clean + prom_gate * detail_2d, base_clean * 0.25) * prom_gate
    return np.clip(prom_detailed, 0.0, None).astype(np.float32)


def calibrate_and_sharpen(
    img: np.ndarray, cx: float, cy: float, r: float, harmonics: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray, float]:
    r_rel = compute_relative_radius(img.shape, cx, cy, r, harmonics)

    # 1. Measure sky noise floor well beyond prominences (> 1.18R)
    sky_mask = (r_rel > 1.18) & (r_rel < 1.45)
    if np.count_nonzero(sky_mask) < 100:
        sky_mask = r_rel > 1.12
    sky_pixels = img[sky_mask]
    sky_med = float(np.median(sky_pixels))
    sky_std = 1.4826 * float(np.median(np.abs(sky_pixels - sky_med)))

    # 2. Lucy-Richardson deconvolution on the UNCLIPPED raw image (sky pedestal intact -> zero step-edge ringing!)
    iters = cfg["sharpening"]["lr_iterations"]
    deconv = img.copy()
    if iters > 0:
        sigma = estimate_seeing_psf(img, cx, cy, r) if cfg["sharpening"]["auto_seeing_psf"] else 1.0
        print(f"  -> L-R deconvolution: PSF sigma={sigma:.2f} px, iterations={iters}")
        size = int(math.ceil(sigma * 6)) | 1
        ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
        gx, gy = np.meshgrid(ax, ax)
        psf = np.exp(-(gx**2 + gy**2) / (2.0 * sigma**2))
        psf /= np.sum(psf)
        m = float(np.max(deconv))
        if m > 0:
            deconv = (skrest.richardson_lucy(deconv / m, psf, num_iter=iters, clip=False) * m).astype(np.float32)

    # 2b. Extract full-detail off-limb prominences from unclipped deconvolved image BEFORE disk floor clipping
    prom_cal = extract_offlimb_prominences(deconv, r_rel, cx, cy, r, harmonics, sky_med, sky_std)

    # 3. Disk calibration: subtract sky floor, normalize to disk peak, flat-field & FNRGF
    floor = sky_med + cfg["calibration"]["sky_clip_sigma"] * sky_std
    out = np.clip(deconv - floor, 0.0, None)
    disk_peak = float(np.percentile(out[r_rel < 0.95], 99.6))
    if disk_peak > 0:
        out /= disk_peak

    # 3. Smooth etalon sweet-spot radial + 2D illumination flat-field inside 0.990R
    # Leaves a natural 3D chromospheric texture and edge relief inside the limb so the outer ring
    # retains crisp spicule definition instead of blowing out into a solid bright band!
    flat_strength = cfg["calibration"]["radial_flat_strength"]
    if flat_strength > 0.0:
        num_bins = 140
        r_edges = np.linspace(0, 0.990, num_bins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        prof = np.array([
            np.percentile(out[(r_rel >= r_edges[i]) & (r_rel < r_edges[i + 1])], 65)
            for i in range(num_bins)
        ], dtype=np.float32)
        prof = ndi.gaussian_filter1d(prof, sigma=5.0)
        center_ref = float(np.median(prof[:12]))
        prof_norm = np.clip(prof / max(center_ref, 1e-4), 0.34, 1.0)
        flat_radial = np.interp(r_rel, r_centers, prof_norm, left=prof_norm[0], right=prof_norm[-1])

        t_flat = np.clip((r_rel - 0.985) / 0.012, 0.0, 1.0)
        limb_taper = 1.0 - t_flat * t_flat * (3.0 - 2.0 * t_flat)
        out = out / (1.0 - (flat_strength * limb_taper) * (1.0 - flat_radial))

        # Gentle large-scale 2D etalon illumination leveler inside 0.95R (sigma ~ 0.16R)
        # Removes off-center etalon sweet-spot and transmission gradients (such as the upper-right
        # dark tuning gradient in 12_32_18_Drizzle15) while preserving active regions and filaments!
        disk_med = float(np.median(out[r_rel < 0.90]))
        clamped_disk = np.where(r_rel < 0.96, np.clip(out, disk_med * 0.86, disk_med * 1.14), disk_med)
        illum_2d = ndi.gaussian_filter(clamped_disk, sigma=max(r * 0.16, 85.0))
        t_2d = np.clip((r_rel - 0.93) / 0.05, 0.0, 1.0)
        w_2d = (1.0 - t_2d * t_2d * (3.0 - 2.0 * t_2d)) * (flat_strength * 0.85)
        out = out * (1.0 + w_2d * ((disk_med / np.maximum(illum_2d, 1e-4)) - 1.0))

    # 3b. Harmonic FNRGF / CNRGF local angular variance equalizer (from SolarFinish / Druckmüller)
    # Uses local angular mean mu(r_rel, theta) directly to level remaining azimuthal etalon drift
    # while adaptively equalizing contrast along constant-r_rel shells!
    fnrgf_min = cfg["calibration"].get("fnrgf_min_gain", 1.35)
    fnrgf_max = cfg["calibration"].get("fnrgf_max_gain", 2.40)
    fnrgf_blend = cfg["calibration"].get("fnrgf_blend", 0.55)
    if fnrgf_blend > 0.0:
        out = apply_harmonic_fnrgf(out, r_rel, cx, cy, min_gain=fnrgf_min, max_gain=fnrgf_max, blend=fnrgf_blend)

    # 4. Resolution-adaptive multi-scale detail refinement (fine spicules + medium fibrils)
    # Scales filter sigmas to the solar disk radius (r / 680.0) so standard stacks (r~680) and
    # 1.5x Drizzle stacks (r~1040) receive identical physical chromospheric sharpness!
    res_scale = float(np.clip(r / 680.0, 0.75, 2.0))
    g1 = ndi.gaussian_filter(out, sigma=1.00 * res_scale)
    g2 = ndi.gaussian_filter(out, sigma=3.00 * res_scale)
    fine_layer = out - g1
    med_layer = g1 - g2

    fine_compressed = 0.032 * np.tanh(fine_layer / 0.032)
    med_compressed = 0.055 * np.tanh(med_layer / 0.055)
    snr_gate = np.clip((out - 0.008) / 0.05, 0.0, 1.0)
    fine_amt = cfg["sharpening"]["fine_detail"] - 1.0
    med_amt = cfg["sharpening"]["medium_detail"] - 1.0
    out = out + snr_gate * (fine_amt * fine_compressed + med_amt * med_compressed)

    return np.clip(out, 0.0, None).astype(np.float32), prom_cal, sky_std


# ==============================================================================
# 4. AUTO-DETECTED 3-ZONE DYNAMIC RANGE STRETCH (DISK, LIMB RING, PROMINENCES)
# ==============================================================================

def auto_stretch_zone(
    signal: np.ndarray,
    zone_mask: np.ndarray,
    q_dark: float,
    q_mid: float,
    q_bright: float,
    target_dark: float,
    target_mid: float,
    target_bright: float,
    min_floor: float,
    max_ceil: float,
    gamma_dark: float = 0.95,
    gamma_bright: float = 0.92,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Automatically detects the darkest (q_dark), median (q_mid), and brightest (q_bright)
    pixels within zone_mask and stretches the signal smoothly and monotonically to
    [target_dark, target_mid, target_bright] with C^1-continuous asymptotic tanh roll-off
    to [min_floor, max_ceil] in the extreme tails so zero pixels ever clip flat.
    """
    vals = signal[zone_mask]
    if len(vals) < 16:
        return np.clip(signal, min_floor, max_ceil), (0.0, 0.5, 1.0)

    s_dark = float(np.percentile(vals, q_dark))
    s_mid = float(np.percentile(vals, q_mid))
    s_bright = float(np.percentile(vals, q_bright))

    # Ensure strict monotonicity of detected anchor points
    if s_mid <= s_dark + 1e-4:
        s_mid = s_dark + 1e-3
    if s_bright <= s_mid + 1e-4:
        s_bright = s_mid + 1e-3

    out = np.empty_like(signal, dtype=np.float32)

    # 1. Dark side (signal <= s_mid): maps [s_dark, s_mid] -> [target_dark, target_mid]
    low_mask = signal <= s_mid
    s_low = signal[low_mask]
    span_dark = max(s_mid - s_dark, 1e-5)
    t_dark = np.maximum((s_mid - s_low) / span_dark, 0.0)

    in_dark = np.minimum(t_dark, 1.0)
    val_dark = target_mid - (target_mid - target_dark) * np.power(in_dark, gamma_dark)

    # C^1 smooth asymptotic tail below s_dark (t_dark > 1.0) down toward min_floor
    tail_headroom = max(target_dark - min_floor, 1e-4)
    slope_dark = gamma_dark * (target_mid - target_dark)
    excess_dark = np.maximum(t_dark - 1.0, 0.0)
    val_dark = val_dark - tail_headroom * np.tanh((slope_dark * excess_dark) / tail_headroom)
    out[low_mask] = val_dark

    # 2. Bright side (signal > s_mid): maps [s_mid, s_bright] -> [target_mid, target_bright]
    high_mask = ~low_mask
    s_high = signal[high_mask]
    span_bright = max(s_bright - s_mid, 1e-5)
    t_bright = np.maximum((s_high - s_mid) / span_bright, 0.0)

    in_bright = np.minimum(t_bright, 1.0)
    val_bright = target_mid + (target_bright - target_mid) * np.power(in_bright, gamma_bright)

    # C^1 smooth asymptotic roll-off above s_bright (t_bright > 1.0) up toward max_ceil
    ceil_headroom = max(max_ceil - target_bright, 1e-4)
    slope_bright = gamma_bright * (target_bright - target_mid)
    excess_bright = np.maximum(t_bright - 1.0, 0.0)
    val_bright = val_bright + ceil_headroom * np.tanh((slope_bright * excess_bright) / ceil_headroom)
    out[high_mask] = val_bright

    return np.clip(out, min_floor, max_ceil), (s_dark, s_mid, s_bright)


def shape_solar_luminance(
    img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    harmonics: np.ndarray,
    cfg: dict,
    verbose: bool = True,
    prom_img: np.ndarray | None = None,
    sky_std: float = 0.0,
) -> np.ndarray:
    r_rel = compute_relative_radius(img.shape, cx, cy, r, harmonics)

    style = cfg["tone"]["style"]
    contrast = cfg["tone"]["contrast"]
    gamma = cfg["tone"]["midtone_gamma"]

    # Target tonal anchors balancing surface darks, outer limb ring, and prominence highlights
    t_disk_dark = cfg["tone"].get("target_disk_dark", 0.14)
    t_disk_mid = cfg["tone"].get("target_disk_mid", 0.49)
    t_disk_bright = cfg["tone"].get("target_disk_bright", 0.75)

    t_ring_dark = cfg["tone"].get("target_ring_dark", 0.22)
    t_ring_mid = cfg["tone"].get("target_ring_mid", 0.51)
    t_ring_bright = cfg["tone"].get("target_ring_bright", 0.72)

    t_prom_bright = cfg["tone"].get("target_prom_bright", 0.64)

    disk_mask = r_rel < 0.89
    ring_mask = (r_rel >= 0.935) & (r_rel <= 0.992)
    d_min = float(np.percentile(img[r_rel < 0.94], 0.5))
    d_med = float(np.percentile(img[disk_mask], 50.0))

    # Build a smooth radial median baseline across the outer disk & limb ring (0.84R -> 0.998R)
    # with fine spacing near the limb so radial_base tracks the optical roll-off cleanly without a 0.991R spike
    r_prof_knots = np.concatenate([
        np.linspace(0.84, 0.96, 19, dtype=np.float32),
        np.linspace(0.962, 0.998, 19, dtype=np.float32),
    ])
    r_prof_vals = np.array([
        float(np.percentile(img[(r_rel >= rk - 0.002) & (r_rel < rk + 0.002)], 50))
        if np.any((r_rel >= rk - 0.002) & (r_rel < rk + 0.002)) else d_med
        for rk in r_prof_knots
    ], dtype=np.float32)
    r_prof_vals = ndi.gaussian_filter1d(r_prof_vals, sigma=0.9, mode="nearest")
    radial_base = np.interp(r_rel, r_prof_knots, r_prof_vals, left=d_med, right=max(r_prof_vals[-1], 1e-4))

    res_s = float(np.clip(r / 680.0, 0.75, 2.0))

    if style in {"limb_glow", "inverted", "hybrid"}:
        blend_amt = (
            1.0
            if style == "inverted"
            else (0.50 if style == "hybrid" else cfg["tone"]["limb_glow_blend"])
        )
        u_disk = img / max(d_med, 1e-4)

        # Detect active-region magnetic complexes via local bright plage envelope (u > 1.08)
        plage_seed = np.clip((u_disk - 1.08) / 0.12, 0.0, 1.0) * (r_rel < 0.96).astype(np.float32)
        ar_zone = np.clip(ndi.gaussian_filter(plage_seed, sigma=max(r * 0.032, 12.0)) * 2.6, 0.0, 1.0)

        # Continuous un-clamped inverted disk signal (bright plages/canopies -> low, dark filaments -> high)
        inv_signal = 2.0 - u_disk

        # Preserve true sunspot umbrae (u_disk < 0.89 inside active regions) as the darkest features
        sunspot_core = ndi.gaussian_filter(ar_zone * np.clip((0.90 - u_disk) / 0.12, 0.0, 1.0), sigma=1.4)
        inv_signal = (1.0 - sunspot_core) * inv_signal + sunspot_core * (0.45 + 0.55 * u_disk)

        raw_disk_signal = (1.0 - blend_amt) * u_disk + blend_amt * inv_signal

        # Outer limb ring signal: normalized against the local radial baseline + crisp positive spicule detail
        u_ring = img / np.maximum(radial_base, 1e-4)
        ring_hf = u_ring - ndi.gaussian_filter(u_ring, sigma=1.6 * res_s)
        inv_ring = 2.0 - u_ring + 0.55 * ring_hf
        raw_ring_signal = (1.0 - blend_amt) * u_ring + blend_amt * inv_ring
    else:
        # Classic natural positive disk ("natural", "natural_disk", "high_contrast"):
        # Preserve gentle monotonic limb darkening into the outer ring (no bright halo ring!)
        if style == "high_contrast":
            contrast = max(contrast, 1.30)
            t_disk_dark = min(t_disk_dark, 0.10)
            t_disk_bright = max(t_disk_bright, 0.80)
        t_ring_dark = min(t_ring_dark, 0.16)
        t_ring_mid = min(t_ring_mid, 0.44)
        t_ring_bright = min(t_ring_bright, 0.68)
        u_disk = img / max(d_med, 1e-4)
        raw_disk_signal = u_disk * (1.0 - 0.14 * (np.clip(r_rel, 0.0, 1.0) ** 2.0))
        u_ring = img / np.maximum(radial_base, 1e-4)
        ring_hf = u_ring - ndi.gaussian_filter(u_ring, sigma=1.6 * res_s)
        raw_ring_signal = u_ring * (1.0 - 0.04 * np.clip((r_rel - 0.88) / 0.11, 0.0, 1.0)) + 0.30 * ring_hf

    # ZONE 1: Auto-detect darkest, median, and brightest surface pixels and stretch smoothly
    disk_stretched, (d_pdark, d_pmid, d_pbright) = auto_stretch_zone(
        raw_disk_signal,
        disk_mask,
        q_dark=0.15,
        q_mid=50.0,
        q_bright=99.85,
        target_dark=t_disk_dark,
        target_mid=t_disk_mid,
        target_bright=t_disk_bright,
        min_floor=0.05,
        max_ceil=0.86,
        gamma_dark=0.92,
        gamma_bright=0.82,
    )

    # ZONE 2: Auto-detect darkest, median, and brightest pixels in the outer limb ring and stretch smoothly
    ring_stretched, (r_pdark, r_pmid, r_pbright) = auto_stretch_zone(
        raw_ring_signal,
        ring_mask,
        q_dark=0.50,
        q_mid=50.0,
        q_bright=99.50,
        target_dark=t_ring_dark,
        target_mid=t_ring_mid,
        target_bright=t_ring_bright,
        min_floor=0.10,
        max_ceil=0.78,
        gamma_dark=0.95,
        gamma_bright=0.90,
    )

    # Smooth C^2 Hermite handoff from inner disk stretch to outer limb ring stretch across 0.88R -> 0.955R
    t_ring = np.clip((r_rel - 0.88) / 0.075, 0.0, 1.0)
    w_ring = t_ring * t_ring * (3.0 - 2.0 * t_ring)
    disk_toned = (1.0 - w_ring) * disk_stretched + w_ring * ring_stretched

    # Taper disk_toned smoothly before the 0.998R limb cliff so inverted ring mapping never spikes into a bright wire
    outer_feather = 1.0 - np.clip((r_rel - 0.992) / 0.007, 0.0, 1.0)
    outer_feather = outer_feather * outer_feather * (3.0 - 2.0 * outer_feather)
    disk_toned = disk_toned * outer_feather

    # Optional fine user contrast & midtone gamma adjustment around the auto-stretched disk
    if abs(contrast - 1.0) > 1e-3:
        x = np.clip(disk_toned, 1e-6, 1.0 - 1e-6)
        disk_toned = 1.0 / (1.0 + np.power((1.0 - x) / x, contrast))

    if abs(gamma - 1.0) > 1e-3:
        g_weight = cfg["tone"].get("gamma_weight", 0.75)
        d_clamped = np.clip(disk_toned, 0.0, 1.0)
        disk_toned = g_weight * np.power(d_clamped, gamma) + (1.0 - g_weight) * (
            1.0 - np.power(1.0 - d_clamped, 1.0 / max(gamma, 0.1))
        )

    # ZONE 3: Off-limb prominence auto-detection & smooth stretch (continuous from 0.00 sky -> target_prom_bright!)
    prom_src = prom_img if prom_img is not None else img
    offlimb_ann = (r_rel >= 1.008) & (r_rel <= 1.120)
    active_prom_mask = offlimb_ann & (prom_src > max(1.5 * sky_std, 1e-4))

    if np.count_nonzero(active_prom_mask) >= 30:
        prom_vals = prom_src[active_prom_mask]
        p_pdark = float(np.percentile(prom_vals, 5.0))
        p_pmid = float(np.percentile(prom_vals, 50.0))
        p_pbright = float(np.percentile(prom_vals, 99.2))
        prom_scale_ref = max(p_pbright, 6.0 * sky_std, 0.035)
        prom_norm = prom_src / prom_scale_ref
        prom_gamma = cfg["tone"].get("prominence_gamma", 0.60)
        # Smooth asymptotic compression: faint outer wisps fade smoothly to 0.00, while bright prominence
        # threads glow clearly at target_prom_bright (~0.64) and the limb transition stays capped near ~0.55
        # so there is zero bright wire or hard posterized border!
        prom_comp = np.clip(prom_norm / (1.0 + 0.42 * prom_norm) * 1.42, 0.0, 1.25)
        prom_toned = t_prom_bright * np.power(prom_comp, prom_gamma)
    else:
        p_pdark, p_pmid, p_pbright = 0.0, 0.0, 0.0
        prom_toned = np.zeros_like(prom_src)

    if verbose:
        print(
            f"  -> Auto-stretch anchors | "
            f"Disk [dark={d_pdark:.3f}, mid={d_pmid:.3f}, bright={d_pbright:.3f}] -> [{t_disk_dark:.2f}, {t_disk_mid:.2f}, {t_disk_bright:.2f}] | "
            f"Ring [dark={r_pdark:.3f}, mid={r_pmid:.3f}, bright={r_pbright:.3f}] -> [{t_ring_dark:.2f}, {t_ring_mid:.2f}, {t_ring_bright:.2f}] | "
            f"Prom [dark={p_pdark:.4f}, mid={p_pmid:.4f}, bright={p_pbright:.4f}] -> [0.00, {t_prom_bright:.2f}]"
        )

    # Seamless crossfade across the physical limb contour (0.992R -> 0.999R); disk_toned already carries (1 - prom_weight)
    t_prom = np.clip((r_rel - 0.992) / 0.007, 0.0, 1.0)
    prom_weight = t_prom * t_prom * (3.0 - 2.0 * t_prom)
    # Match prom_toned's limb-crossing brightness to the outer ring (~t_ring_mid) across 0.992R -> 1.004R
    # so the disk rolls organically into spicules with zero bright rim wire in any style
    t_relax = np.clip((r_rel - 0.999) / 0.005, 0.0, 1.0)
    w_relax = t_relax * t_relax * (3.0 - 2.0 * t_relax)
    prom_at_limb = np.minimum(prom_toned, t_ring_mid * 1.04)
    prom_toned = (1.0 - w_relax) * prom_at_limb + w_relax * prom_toned
    combined = disk_toned + prom_weight * prom_toned

    return np.clip(combined, 0.0, 1.0).astype(np.float32)


# ==============================================================================
# 5. INCANDESCENT PER-CHANNEL SOLAR COLOR SOLVER (GOLDEN HIGHLIGHTS)
# ==============================================================================

def hex_to_lab(hex_str: str) -> np.ndarray:
    hex_str = hex_str.lstrip("#")
    rgb = np.array([int(hex_str[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
    return skcolor.rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)


def colorize_solar_incandescent(
    gray: np.ndarray, cx: float, cy: float, r: float, cfg: dict
) -> np.ndarray:
    """
    Solves for smooth per-channel power curves anchored to target_disk_hex at the
    mid-disk median. Unlike flat CIELAB LUTs, per-channel power curves naturally
    shift bright limb/filaments into rich, saturated golden-yellow while keeping
    prominence wisps fiery orange-red.
    """
    target_lab = hex_to_lab(cfg["color"]["target_disk_hex"])
    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    sample = gray[(rr >= 0.15 * r) & (rr <= 0.85 * r)]
    if len(sample) > 5000:
        sample = sample[:: len(sample) // 5000]

    def loss(gammas):
        gr, gg, gb = gammas
        rgb_s = np.stack(
            [np.power(sample, gr), np.power(sample, gg), np.power(sample, gb)], axis=-1
        ).reshape(1, -1, 3)
        lab_s = skcolor.rgb2lab(rgb_s).reshape(-1, 3)
        med = np.median(lab_s, axis=0)
        # Keep Red channel exponent gr >= 0.50 (like SolarFinish's 0.50, 1.25, 3.75) so dark
        # fibrils and magnetic canopies render as deep, dark bronze-red rather than flat bright red!
        return (
            0.8 * (med[0] - target_lab[0]) ** 2
            + 2.5 * (med[1] - target_lab[1]) ** 2
            + 2.5 * (med[2] - target_lab[2]) ** 2
            + 25.0 * max(0.0, 0.50 - gr) ** 2
        )

    res = minimize(
        loss,
        x0=[0.52, 1.32, 3.9],
        bounds=[(0.48, 2.2), (0.6, 4.0), (1.8, 9.0)],
        method="L-BFGS-B",
    )
    gr, gg, gb = res.x

    rgb = np.dstack([
        np.power(gray, gr),
        np.power(gray, gg),
        np.power(gray, gb),
    ])

    sky_warmth = cfg["color"]["sky_warmth"]
    if sky_warmth > 0.0:
        # Screen-blend a deep H-alpha crimson sky floor (~4.5:1 R:G ratio) so dark sky
        # lifts to a rich reddish-amber glow without clipping or washing out bright disk highlights
        rgb[..., 0] = rgb[..., 0] + sky_warmth * (1.0 - rgb[..., 0])
        rgb[..., 1] = rgb[..., 1] + (sky_warmth * 0.22) * (1.0 - rgb[..., 1])

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ==============================================================================
# 6. 3x3 PREVIEW GRID (COMPARING STYLES & COLOR TEMPERATURES)
# ==============================================================================

def build_preview_grid(
    calibrated_img: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    harmonics: np.ndarray,
    cfg: dict,
    out_path: Path,
    prom_img: np.ndarray | None = None,
    sky_std: float = 0.0,
):
    """
    Generates a 3x3 comparison sheet:
      Row 1: AstroBin "limb_glow" style in Deep Fiery Orange (#E64800)
      Row 2: AstroBin "limb_glow" style in Golden Solar Amber (#F58200)
      Row 3: Classic "natural_disk" style in Rich Burnt Orange (#EB5E00)
      Columns: Contrast 1.05 | 1.22 | 1.38
    """
    scale = min(1.0, 512.0 / max(calibrated_img.shape))
    small = (
        sktrans.rescale(calibrated_img, scale, anti_aliasing=True, preserve_range=True).astype(np.float32)
        if scale < 1.0
        else calibrated_img
    )
    small_prom = (
        sktrans.rescale(prom_img, scale, anti_aliasing=True, preserve_range=True).astype(np.float32)
        if (prom_img is not None and scale < 1.0)
        else prom_img
    )
    scx, scy, sr = cx * scale, cy * scale, r * scale
    sharm = harmonics * scale

    row_specs = [
        ("limb_glow", "#E64800"),
        ("limb_glow", "#F58200"),
        ("natural_disk", "#EB5E00"),
    ]
    contrasts = [1.05, 1.22, 1.38]

    rows = []
    for style_mode, hex_col in row_specs:
        row_imgs = []
        for c_val in contrasts:
            temp_cfg = {
                "tone": {**cfg["tone"], "style": style_mode, "contrast": c_val},
                "color": {**cfg["color"], "target_disk_hex": hex_col},
            }
            g_tile = shape_solar_luminance(
                small, scx, scy, sr, sharm, temp_cfg, verbose=False, prom_img=small_prom, sky_std=sky_std
            )
            c_tile = colorize_solar_incandescent(g_tile, scx, scy, sr, temp_cfg)
            c_tile[:2, :] = 0.25
            c_tile[:, :2] = 0.25
            row_imgs.append(c_tile)
        rows.append(np.concatenate(row_imgs, axis=1))

    grid = np.concatenate(rows, axis=0)
    _safe_write(iio.imwrite, out_path, (np.clip(grid, 0, 1) * 255).astype(np.uint8))


# ==============================================================================
# 7. CENTERING, CROPPING & MAIN FLOW
# ==============================================================================

def center_and_crop_square(
    img: np.ndarray, cx: float, cy: float, r: float, crop_radii: float
) -> tuple[np.ndarray, float, float]:
    half_size = int(round(r * crop_radii))
    full_size = half_size * 2

    pad_left = max(0, int(math.ceil(half_size - cx)))
    pad_right = max(0, int(math.ceil(cx + half_size - img.shape[1])))
    pad_top = max(0, int(math.ceil(half_size - cy)))
    pad_bottom = max(0, int(math.ceil(cy + half_size - img.shape[0])))

    padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    x0 = int(round(cx + pad_left - half_size))
    y0 = int(round(cy + pad_top - half_size))

    return padded[y0 : y0 + full_size, x0 : x0 + full_size], float(half_size), float(half_size)


def process_single_file(filepath: Path, out_dir: Path, cfg: dict, style_tag: str | None = None):
    print(f"[SolarAuto v{__version__}] Processing: {filepath.name}")
    img, _ = load_solar_image(filepath)

    geom = cfg["geometry"]
    if geom["flip_horizontal"]:
        img = np.fliplr(img)
    if geom["flip_vertical"]:
        img = np.flipud(img)
    if abs(geom["manual_rotate_deg"]) > 1e-3:
        img = sktrans.rotate(img, geom["manual_rotate_deg"], resize=False, preserve_range=True)

    if "_imppg" in filepath.stem.lower() and cfg["sharpening"]["lr_iterations"] > 0:
        print("  -> Detected '_imppg' in filename: skipping double L-R deconvolution.")
        cfg = {**cfg, "sharpening": {**cfg["sharpening"], "lr_iterations": 0}}

    cx, cy, r, harmonics = find_subpixel_limb(img)
    print(
        f"  -> Limb locked: center=({cx:.2f}, {cy:.2f}), radius={r:.2f} px, "
        f"ellipticity=({harmonics[2]:+.2f}, {harmonics[3]:+.2f}) px"
    )

    img_cal, prom_cal, sky_std = calibrate_and_sharpen(img, cx, cy, r, harmonics, cfg)

    if geom["auto_center"]:
        img_cal, cx_c, cy_c = center_and_crop_square(
            img_cal, cx, cy, r, geom["crop_solar_radii"]
        )
        prom_cal, _, _ = center_and_crop_square(
            prom_cal, cx, cy, r, geom["crop_solar_radii"]
        )
        cx, cy = cx_c, cy_c

    style_suffix = f"-{style_tag}" if style_tag else ""
    gray_final = shape_solar_luminance(
        img_cal, cx, cy, r, harmonics, cfg, prom_img=prom_cal, sky_std=sky_std
    )
    color_final = colorize_solar_incandescent(gray_final, cx, cy, r, cfg)
    save_outputs(out_dir, filepath.stem, gray_final, color_final, cfg, style_suffix=style_suffix)

    if cfg["pipeline"]["generate_preview_grid"]:
        grid_path = out_dir / f"{filepath.stem}_preview_grid{style_suffix}.png"
        build_preview_grid(
            img_cal, cx, cy, r, harmonics, cfg, grid_path, prom_img=prom_cal, sky_std=sky_std
        )
        print(f"  -> Saved 3x3 style & color grid: {grid_path.name}")


def main():
    default_profile = Path(__file__).resolve().with_name("solar_profile.toml")
    style_choices = ["limb_glow", "inverted", "hybrid", "natural", "natural_disk", "high_contrast"]
    parser = argparse.ArgumentParser(
        description="Reference-grade automated solar image processor.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument(
        "-p",
        "--profile",
        type=Path,
        default=default_profile,
        help=f"Optional TOML profile override (default: {default_profile.name} next to solar_auto.py)",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("./output"), help="Output directory (default: ./output)")
    parser.add_argument(
        "-s",
        "--style",
        type=str,
        choices=style_choices,
        default=None,
        help=(
            "Override disk inversion & contrast style from the profile:\n"
            "  limb_glow     - AstroBin-style 88%% inverted disk, golden filaments, dark canopies, bright limb & proms\n"
            "  inverted      - Full 100%% inverted disk (bright gold filaments/fibrils, deep crimson plages, positive proms)\n"
            "  hybrid        - Balanced 50/50 positive/inverted disk (tames bright active regions while keeping limb glow)\n"
            "  natural       - Traditional positive chromosphere (dark filaments, bright white-gold plages, positive proms)\n"
            "  natural_disk  - Alias for 'natural'\n"
            "  high_contrast - Dramatic positive chromosphere with deeper dark filaments and punchier plages"
        ),
    )
    parser.add_argument("--grid", action="store_true", help="Also save a 3x3 style & color comparison grid")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    with open(args.profile, "rb") as f:
        cfg = tomllib.load(f)
    if args.style is not None:
        cfg["tone"]["style"] = args.style
    if args.grid:
        cfg["pipeline"]["generate_preview_grid"] = True

    if args.input.is_dir():
        exts = {".tif", ".tiff", ".fit", ".fits", ".png"}
        for fp in sorted(p for p in args.input.iterdir() if p.suffix.lower() in exts):
            process_single_file(fp, args.output, cfg, style_tag=args.style)
    else:
        process_single_file(args.input, args.output, cfg, style_tag=args.style)


if __name__ == "__main__":
    main()