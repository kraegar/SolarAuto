# Changelog

All notable changes to **SolarAuto** are documented in this file.
Versioning follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

---

## [1.0.0] — 2026-11-20

Initial public release of the autonomous 16-bit Hydrogen-Alpha full-disk solar processor.

### Processing & Calibration
- **Sub-Pixel 2nd-Order Harmonic Limb Lock (`find_subpixel_limb`):** Deterministic RANSAC circle initialization (`rng=42`) followed by two 72-sector sub-pixel passes at the 45% optical limb threshold, fitting 2nd-order Fourier ellipticity harmonics to lock `r_rel = 1.000` within ~0.3 px across all 360° (with tight-crop fallback).
- **Seeing-Adaptive Unclipped Deconvolution (`estimate_seeing_psf`):** Scales the limb sample window by `r / 680.0`, subtracts intrinsic chromospheric spicule-skirt width (`1.15 * res_s`) in quadrature, and runs 16-iteration Lucy-Richardson deconvolution on the unclipped stack with its sky pedestal intact to avoid prominence step-edge ringing.
- **Non-Destructive Off-Limb Prominence Extraction (`extract_offlimb_prominences`):** Models smooth atmospheric sky scatter outside `1.028R` via a 72° polar minimum filter (monotonic across `1.028R–1.18R`, clamped to `sky_med`, held flat inward) and uses inward-ray clamping from `1.010R` for 4-scale Gaussian prominence wavelets (`0.75–7.00 * res_s`) with 2 px border suppression.
- **Etalon Flat-Fielding & Direct-Polar Harmonic FNRGF (`apply_harmonic_fnrgf`):** Removes radial and 2D sweet-spot gradients, equalizes local angular mean and standard deviation across 160 × 72 `(r_rel, theta)` shells, and applies 2-scale `tanh`-limited disk wavelets.

### Tone Shaping, Limb Transition & Color
- **3-Zone Auto-Detected Dynamic Range Stretch (`auto_stretch_zone`, `shape_solar_luminance`):** Independently detects dark/mid/bright percentiles across the inner disk (`r < 0.89R`), outer limb ring (`0.935R–0.992R`), and off-limb prominences (`1.008R–1.12R`).
- **Wire-Free Limb Crossfade & Reduced Rim:** Refined `0.962R–0.998R` radial baseline sampling, applied the `prom_at_limb` limb-crossing ceiling across all styles, and lowered `target_ring_mid` (`0.51`) and `target_ring_bright` (`0.70`) to eliminate the `0.990R` rim spike artifact.
- **Monotonic Limb Darkening in Positive Modes:** In `natural` and `high_contrast` modes, capped outer-ring targets (`<= [0.16, 0.44, 0.68]`) to preserve natural center-to-limb darkening without a bright outer halo.
- **Incandescent Color Solver & Reddish Sky Glow (`colorize_solar_incandescent`):** `L-BFGS-B` CIELAB solver anchored to `target_disk_hex = "#E66D00"` with Red-channel floor `gr >= 0.50` for deep bronze-crimson shadows, plus screen-blended `sky_warmth = 0.14` (`~4.5:1` R:G) for a natural deep crimson sky background without washing out disk highlights.

### CLI & Packaging
- **Default Profile & `--style` Flag:** Defaults `-p/--profile` to `solar_profile.toml` next to `solar_auto.py`, adds `-s/--style` (`limb_glow`, `inverted`, `hybrid`, `natural`, `natural_disk`, `high_contrast`) with `-h` help, appends `-<style>` to output filenames only when `--style` is specified, and generates the 3×3 preview grid only when `--grid` is passed.
- **Version & Documentation:** Added `__version__ = "1.0.0"`, `-V/--version` CLI flag, `SolarAuto_Pipeline_Technical_Summary.pdf`, `README.md`, MIT `LICENSE`, `requirements.txt`, and `.gitignore`.
- **Stack Evaluator (`evaluate.py`):** Included `evaluate.py` (ported to `scipy.ndimage` / `scikit-image` to match `requirements.txt`) to rank AutoStakkert!4 `.tif`/`.tiff` stacks by Sobel focus, Immerkær SNR, etalon contrast, and wind isotropy.
