# SolarAuto

Autonomous, reference-grade Python processor for 16-bit Hydrogen-Alpha (H-α) full-disk solar stacks.

SolarAuto takes raw stacked solar images (TIFF, FITS, or PNG) and automatically performs sub-pixel harmonic limb locking, seeing-adaptive Lucy-Richardson deconvolution, non-destructive off-limb prominence extraction, direct-polar FNRGF contrast equalization, 3-zone dynamic range stretching, and CIELAB-calibrated incandescent colorization.

For a deep-dive technical explanation of the algorithms and math behind each stage, see [`SolarAuto_Pipeline_Technical_Summary.pdf`](SolarAuto_Pipeline_Technical_Summary.pdf).

---

## Features

- **2nd-Order Harmonic Limb Lock:** Sub-pixel RANSAC + Fourier harmonic contour fit locks $r_{\text{rel}} = 1.000$ around all $360^\circ$, handling atmospheric refraction and optical tilt.
- **Seeing-Adaptive Unclipped Deconvolution:** Estimates the optical seeing PSF from the limb profile (removing intrinsic spicule-skirt width in quadrature) and runs Lucy-Richardson deconvolution without clipping the sky pedestal.
- **Inward-Ray-Clamped Prominence Wavelets:** Separates smooth polar atmospheric scatter and applies 4-scale wavelets without bright-disk bleed, preserving fine vertical prominence hairs and dark cavities against a pure black sky.
- **Direct-Polar Harmonic FNRGF:** Equalizes off-center etalon sweet-spot gradients and local chromospheric micro-contrast along constant-$r_{\text{rel}}$ shells.
- **3-Zone Auto-Stretch:** Automatically detects darkest, median, and brightest percentiles independently across the inner disk, outer limb ring, and off-limb prominences.
- **Incandescent Per-Channel Color Solver:** Optimizes $(R^{g_R}, G^{g_G}, B^{g_B})$ in CIELAB space anchored to `target_disk_hex` (`#E66D00` by default).

---

## Installation

Python 3.10+ is recommended.

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

Process a single file using the default `solar_profile.toml`:

```bash
python solar_auto.py your_solar_stack.tif
```

Process an entire directory of `.tif` / `.fits` / `.png` stacks:

```bash
python solar_auto.py ./raw_stacks -o ./output
```

### CLI Options

- `-p, --profile <file.toml>` — Override the default `solar_profile.toml`.
- `-o, --output <dir>` — Output directory (default: `./output`).
- `-s, --style <style>` — Override the disk polarity/contrast style and append `-<style>` to output filenames:
  - `limb_glow` *(default)* — 88% partial-inverted disk with bright golden filaments, dark bronze-red magnetic canopies, glowing 3D limb, and positive prominences.
  - `inverted` — Full 100% inverted disk (bright gold filaments/fibrils, deep crimson plages, positive prominences).
  - `hybrid` — Balanced 50/50 positive/inverted disk.
  - `natural` *(alias: `natural_disk`)* — Traditional positive chromosphere (dark filaments, bright white-gold plages, dark sunspots, monotonic limb darkening, positive prominences).
  - `high_contrast` — Dramatic positive chromosphere with deeper dark filaments and punchier plages.
- `--grid` — Also generate a $3 \times 3$ style, contrast, and color-temperature preview grid.

Run `python solar_auto.py -h` to view full help.

---

## Evaluating AutoStakkert!4 Stacks (`evaluate.py`)

Before processing, you can rank a folder of stacked full-disk `.tif`/`.tiff` files from AutoStakkert!4 to pick the sharpest, lowest-noise capture:

```bash
python evaluate.py ./raw_stacks
# Or search subdirectories recursively:
python evaluate.py ./raw_stacks -r
```

`evaluate.py` measures interior disk **Focus** (Sobel spatial gradient), **Noise / SNR** (Immerkær variance kernel), **Etalon Contrast** (flattened 5th–95th percentile spread), and **Wind Isotropy** (X/Y gradient balance), printing a ranked table sorted by resolution and composite score.

---

## Inspiration & Acknowledgements

- **[SolarFinish](https://github.com/galt68/SolarFinish)** by **Greg Alt** — Inspiration for the automated full-disk H-α workflow, Convolutional Normalizing Radial Graded Filter (CNRGF) concept, bimodal gamma brightening, and per-channel power-curve solar colorization.
- **Hana Druckmüllerová** — Originator of the **Fourier Normalizing Radial Graded Filter (FNRGF)** technique on which polar angular mean/standard-deviation solar equalization is based.

---

## License

Released under the [MIT License](LICENSE).
