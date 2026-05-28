"""
example_cs165cu.py
==================
Bayesian ratio pyrometry using the Thorlabs CS165CU camera **without an IR
filter**.

Procedure
---------
1. Load per-channel QE from the xlsx workbook.
2. Build a RatioPyrometer.  LUTs for all three RGB pairs are computed
   automatically:  (blue, red), (green, red), (green, blue).
3. Analyse a region-of-interest in the test image.  The library assumes
   the ROI is isothermal and combines every (pixel, pair) measurement
   into a single posterior on T* using a Gaussian noise model and a
   uniform prior.
4. Print summary statistics (MAP, mean ± std, 68 / 95 % credible
   intervals) and produce a single 3 × 3 figure:

    | row pair      | col 0 (ratio image + ROI) | col 1 (ROI histogram + LUT) | col 2 (posterior update) |
    | blue/red      | ratio map                  | hist & f_br(T) on twin y    | π0 (dashed) → π1 (solid) |
    | green/red     | ratio map                  | hist & f_gr(T) on twin y    | π1 → π2                  |
    | green/blue    | ratio map                  | hist & f_gb(T) on twin y    | π2 → π3 (MAP & 68% CI)   |

Edit the ROI tuple to surround the hot region in your image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import openpyxl

# Make `pyrometer` importable regardless of cwd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from pyrometer import RatioPyrometer       # noqa: E402
from pyrometer.image_io import load_image  # noqa: E402


XLSX_PATH = HERE / "cs165cu_quantum_efficiency.xlsx"
IMAGE_PATH = HERE / "test.ppm"


# ---------------------------------------------------------------------------
# 1.  Load QE data
# ---------------------------------------------------------------------------

wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
ws = wb["Quantum Efficiency"]
rows = [r for r in ws.iter_rows(min_row=4, values_only=True)
        if isinstance(r[2], (int, float))]
wl_nm = np.array([r[2] for r in rows], dtype=float)   # 400–1000 nm, 5 nm step
qe_b  = np.array([r[3] for r in rows], dtype=float)   # %
qe_g  = np.array([r[4] for r in rows], dtype=float)   # %
qe_r  = np.array([r[5] for r in rows], dtype=float)   # %

print(f"QE loaded: {len(wl_nm)} points, {wl_nm[0]:.0f}–{wl_nm[-1]:.0f} nm")


# ---------------------------------------------------------------------------
# 2.  Build the pyrometer  (LUTs for all 3 pairs are computed automatically)
# ---------------------------------------------------------------------------

pyro = RatioPyrometer(
    wavelengths_nm=wl_nm,
    qe_channels={"red": qe_r, "green": qe_g, "blue": qe_b},
    t_range=(500.0, 2500.0),
    n_temps=1000,
    # ir_filter=None  ← default: full transmission (filter physically removed)
)

print("\nLUTs (in sequential-update order):")
for ci, cj in pyro.pairs:
    R = pyro.ratio_lut[(ci, cj)]
    print(f"  {ci}/{cj:<5s}  {pyro.monotonic_direction[(ci, cj)]:<14s}  "
          f"LUT range [{R.min():.4f}, {R.max():.4f}]")


# ---------------------------------------------------------------------------
# 3.  Analyse the ROI
# ---------------------------------------------------------------------------

ROI = (600, 830, 800, 1000)   # (row0, row1, col0, col1) — edit to your hot region

result = pyro.analyze(
    image=IMAGE_PATH,
    roi=ROI,
    channel_order=("red", "green", "blue"),
    wb_red=1.0,
    wb_blue=1.0,
    # CS165CU 8-bit defaults already set: read_noise_dn=0.86, gain=2.72, max_dn=255
)

print(f"\nROI {ROI}  ({ROI[1] - ROI[0]} × {ROI[3] - ROI[2]} px)")
for pair in pyro.pairs:
    print(f"  {pair[0]}/{pair[1]:<5s}: "
          f"{result.n_valid_pixels_per_pair[pair]:6d} valid pixels")

print()
print(f"MAP temperature      : {result.map_temperature:7.1f} K  "
      f"({result.map_temperature - 273.15:.1f} °C)")
print(f"Posterior mean ± std : {result.mean_temperature:7.1f} ± "
      f"{result.std_temperature:.1f} K")
print(f"68% credible interval: [{result.ci_68[0]:.1f}, {result.ci_68[1]:.1f}] K")
print(f"95% credible interval: [{result.ci_95[0]:.1f}, {result.ci_95[1]:.1f}] K")

print(f"\nGoodness of fit at joint MAP (χ²_red ≈ 1 ⇒ model fits the noise):")
for pair in pyro.pairs:
    chi2 = result.chi2_per_pair[pair]
    chi2r = result.reduced_chi2_per_pair[pair]
    n = result.n_valid_pixels_per_pair[pair]
    print(f"  {pair[0]}/{pair[1]:<5s}: "
          f"χ² = {chi2:11.3e},  N = {n:6d},  χ²/N = {chi2r:.2f}")
print(f"  total       : χ²/dof = {result.reduced_chi2_total:.2f}")


# ---------------------------------------------------------------------------
# 4.  Single figure: 3 rows × 3 columns
# ---------------------------------------------------------------------------

img = load_image(IMAGE_PATH)
r0, r1, c0, c1 = ROI
T_grid = result.temperatures_grid

# ---------------------------------------------------------------------------
# Pre-compute common axis limits so every row uses the same scale.
# ---------------------------------------------------------------------------

# (Col 0) Common color limits for the ratio images — based on the pooled
#     percentiles of all valid ratio pixels across the three pairs.
_all_ratio_pixels = np.concatenate([
    result.ratio_images[p][np.isfinite(result.ratio_images[p])]
    for p in pyro.pairs
])
ratio_vmin = float(np.percentile(_all_ratio_pixels, 2))
ratio_vmax = float(np.percentile(_all_ratio_pixels, 98))

# (Col 1) Fixed ratio axis covering the physically interesting range.
hist_xlim = (0.0, 2.0)

# Shared histogram bin edges → max pixel-count across all three
HIST_BINS = np.linspace(hist_xlim[0], hist_xlim[1], 61)
_max_count = 0
for p in pyro.pairs:
    roi_vals = result.ratio_images[p][r0:r1, c0:c1]
    rv = roi_vals[np.isfinite(roi_vals)].ravel()
    if rv.size:
        _max_count = max(_max_count, int(np.histogram(rv, bins=HIST_BINS)[0].max()))
hist_ylim = (0, _max_count * 1.1)

# Shared temperature axis (both Col 1 twin-y and Col 2 x)
T_axis_lim = (float(T_grid.min()), float(T_grid.max()))

# (Col 2) Per-pair log-likelihoods — share both the T-axis range (chosen to
#     bracket every pair's peak) and the log-L y-axis range across rows.
_pair_peak_Ts = [
    float(T_grid[int(np.argmax(result.pair_log_likelihoods[p]))])
    for p in pyro.pairs
]
loglik_xlim = (
    min(_pair_peak_Ts) - 200.0,
    max(_pair_peak_Ts) + 200.0,
)
loglik_ylim = (-3000.0, 100.0)

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for row_idx, pair in enumerate(pyro.pairs):
    ci, cj = pair
    label = f"{ci}/{cj}"
    ratio_full = result.ratio_images[pair]
    ratio_roi = ratio_full[r0:r1, c0:c1]
    f_T = pyro.ratio_lut[pair]
    T_pair = pyro.temperatures_lut[pair]

    ax_img, ax_hist, ax_loglik = axes[row_idx]

    # --- Column 0: full-image ratio map with ROI rectangle (common colormap) ---
    im = ax_img.imshow(ratio_full, cmap="viridis",
                        vmin=ratio_vmin, vmax=ratio_vmax)
    plt.colorbar(im, ax=ax_img, label=label, fraction=0.04, pad=0.03)
    rect = mpatches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        linewidth=1.5, edgecolor="cyan", facecolor="none",
    )
    ax_img.add_patch(rect)
    ax_img.set_title(f"{label} ratio image  (ROI in cyan)")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # --- Column 1: ROI ratio histogram + LUT on twin y-axis (common axes) ---
    roi_values = ratio_roi[np.isfinite(ratio_roi)].ravel()
    if roi_values.size:
        ax_hist.hist(
            roi_values, bins=HIST_BINS, color="steelblue", alpha=0.75,
        )
    ax_hist.set_xlim(hist_xlim)
    ax_hist.set_ylim(hist_ylim)
    ax_hist.set_xlabel(f"{label} ratio")
    ax_hist.set_ylabel("Pixel count", color="steelblue")
    ax_hist.tick_params(axis="y", labelcolor="steelblue")
    # Shade this pair's LUT range
    ax_hist.axvspan(float(f_T.min()), float(f_T.max()),
                    color="green", alpha=0.10, label="LUT range")

    # Twin y-axis: LUT curve (ratio on shared x, temperature on right y)
    ax_lut = ax_hist.twinx()
    ax_lut.plot(f_T, T_pair, color="red", lw=1.5)
    ax_lut.set_ylim(T_axis_lim)
    ax_lut.set_ylabel("Temperature (K)", color="red")
    ax_lut.tick_params(axis="y", labelcolor="red")

    n_valid = result.n_valid_pixels_per_pair[pair]
    ax_hist.set_title(f"{label} ROI histogram + LUT  ({n_valid} valid px)")

    # --- Column 2: per-pair log-likelihood (common T and y limits) ---
    log_L = result.pair_log_likelihoods[pair]
    log_L_shifted = log_L - log_L.max()
    T_peak_pair = float(T_grid[int(np.argmax(log_L))])

    ax_loglik.plot(T_grid, log_L_shifted, color="darkorange", lw=1.5,
                   label=label)
    ax_loglik.axvline(T_peak_pair, color="darkorange", lw=1.0, alpha=0.6,
                      linestyle="--",
                      label=f"peak = {T_peak_pair:.1f} K")
    # Also mark the joint MAP for reference
    ax_loglik.axvline(result.map_temperature, color="red", lw=1.0, alpha=0.6,
                      label=f"joint MAP = {result.map_temperature:.0f} K")
    ax_loglik.set_xlim(loglik_xlim)
    ax_loglik.set_ylim(loglik_ylim)
    ax_loglik.set_xlabel("Temperature (K)")
    ax_loglik.set_ylabel(r"$\log \mathcal{L}_{ij}(T) - \max$")
    ax_loglik.set_title(f"{label} log-likelihood")
    ax_loglik.grid(True, alpha=0.3)
    ax_loglik.legend(fontsize=8, loc="lower right")

fig.suptitle(
    f"CS165CU Bayesian ratio pyrometry  ·  "
    f"MAP T = {result.map_temperature:.0f} K  ·  "
    f"68% CI [{result.ci_68[0]:.0f}, {result.ci_68[1]:.0f}] K",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(HERE / "pyrometry_result.png", dpi=150)
print(f"\nFigure saved → {HERE / 'pyrometry_result.png'}")


# ---------------------------------------------------------------------------
# 5.  Second figure: final log-posterior π_3(T)
# ---------------------------------------------------------------------------

print(f"\nPer-pair log-likelihood peak temperatures:")
for pair in pyro.pairs:
    log_L = result.pair_log_likelihoods[pair]
    T_peak = float(T_grid[int(np.argmax(log_L))])
    print(f"  {pair[0]}/{pair[1]:<5s}: {T_peak:7.2f} K")

fig2, ax2 = plt.subplots(figsize=(9, 5))
log_pi_final = np.log(np.maximum(result.posterior, 1e-300))
log_pi_final = log_pi_final - log_pi_final.max()    # peak at 0
ax2.plot(T_grid, log_pi_final, color="darkorange", lw=1.8)
ax2.axvline(result.map_temperature, color="red", lw=1.0,
            label=f"MAP = {result.map_temperature:.1f} K")
ax2.axvspan(result.ci_68[0], result.ci_68[1],
            color="red", alpha=0.15,
            label=f"68% CI [{result.ci_68[0]:.1f}, {result.ci_68[1]:.1f}] K")
ax2.set_xlim(T_grid.min(), T_grid.max())
ax2.set_ylim(-3000.0, 100.0)
ax2.set_xlabel("Temperature (K)")
ax2.set_ylabel(r"$\log \pi_3(T) - \max$")
ax2.set_title(
    f"Final log-posterior after fusing all three pairs  ·  "
    f"σ = {result.std_temperature:.1f} K (statistical)"
)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower right", fontsize=9)
fig2.tight_layout()
fig2.savefig(HERE / "final_posterior.png", dpi=150)
print(f"Figure saved → {HERE / 'final_posterior.png'}")


# ---------------------------------------------------------------------------
# 6.  Third figure: ROI-only ratio images (autoscaled per pair)
# ---------------------------------------------------------------------------

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
for ax3, pair in zip(axes3, pyro.pairs):
    ci, cj = pair
    label = f"{ci}/{cj}"
    ratio_roi = result.ratio_images[pair][r0:r1, c0:c1]
    valid = ratio_roi[np.isfinite(ratio_roi)]
    if valid.size:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
    else:
        vmin, vmax = 0.0, 1.0
    im3 = ax3.imshow(ratio_roi, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im3, ax=ax3, label=label, fraction=0.046, pad=0.04)
    ax3.set_title(f"{label}  ROI ratio   "
                  f"(min {valid.min():.3f}, max {valid.max():.3f})"
                  if valid.size else f"{label} ROI ratio")
    ax3.set_xticks([])
    ax3.set_yticks([])

fig3.suptitle(
    f"ROI-only ratio images (each auto-scaled).  "
    f"Spatial variations here imply spatial T gradients.",
    fontsize=11,
)
fig3.tight_layout()
fig3.savefig(HERE / "roi_ratio_images.png", dpi=150)
print(f"Figure saved → {HERE / 'roi_ratio_images.png'}")


if "--no-show" not in sys.argv:
    plt.show()
