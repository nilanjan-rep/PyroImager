"""
plotting.py -- Visualization helpers for pyrometer analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def plot_analysis(
    result,
    pyro,
    image: Union[str, Path, np.ndarray],
    roi: Optional[Tuple[int, int, int, int]] = None,
    figsize: Tuple[float, float] = (16, 12),
):
    """
    Generate the standard 3 x 3 analysis figure for a ratio-pyrometry result.

    Layout::

        Row 0 (blue/red)   | Col 0: full-image ratio map + ROI box
                           | Col 1: ROI histogram + LUT on twin y-axis
                           | Col 2: per-pair log-likelihood curve
        Row 1 (green/red)  | ...
        Row 2 (green/blue) | ...

    All ratio images share the same colour scale (2nd-98th percentile of
    valid pixels pooled across all pairs).  All histograms share the same
    ratio axis (0-2) and pixel-count axis.  All log-likelihood panels share
    the same temperature and y-axis limits.

    Parameters
    ----------
    result : PyrometerResult
        Output of :meth:`RatioPyrometer.analyze`.
    pyro : RatioPyrometer
        The pyrometer instance used to produce `result`.
    image : str, Path, or ndarray
        The image that was analysed (path or (H, W, 3) float64 array in [0,1]).
        Used to determine image dimensions for the full-image ratio maps.
    roi : (row0, row1, col0, col1) or None
        Same ROI passed to :meth:`analyze`.  Drawn as a rectangle on the
        ratio-image panels.  None means the full image.
    figsize : (float, float), default (16, 12)
        Matplotlib figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from .image_io import load_image

    if isinstance(image, (str, Path)):
        img = load_image(image)
    else:
        img = np.asarray(image, dtype=float)

    H, W = img.shape[:2]
    if roi is None:
        r0, r1, c0, c1 = 0, H, 0, W
    else:
        r0, r1, c0, c1 = roi

    T_grid = result.temperatures_grid

    # Shared colour scale for all ratio images: pool finite pixels across all pairs
    _all_ratio_pixels = np.concatenate([
        result.ratio_images[p][np.isfinite(result.ratio_images[p])]
        for p in pyro.pairs
    ])
    if _all_ratio_pixels.size:
        ratio_vmin = float(np.percentile(_all_ratio_pixels, 2))
        ratio_vmax = float(np.percentile(_all_ratio_pixels, 98))
    else:
        ratio_vmin, ratio_vmax = 0.0, 1.0

    # Fixed ratio axis for all histograms: 0 to 2
    hist_xlim = (0.0, 2.0)
    HIST_BINS = np.linspace(hist_xlim[0], hist_xlim[1], 61)

    # Shared histogram y-axis: maximum pixel count across all three pairs
    _max_count = 0
    for p in pyro.pairs:
        roi_vals = result.ratio_images[p][r0:r1, c0:c1]
        rv = roi_vals[np.isfinite(roi_vals)].ravel()
        if rv.size:
            _max_count = max(_max_count, int(np.histogram(rv, bins=HIST_BINS)[0].max()))
    hist_ylim = (0, max(_max_count * 1.1, 1))

    # Shared temperature axis (LUT twin-y and log-likelihood x)
    T_axis_lim = (float(T_grid.min()), float(T_grid.max()))

    # Shared log-likelihood x-axis: bracket all pair peaks by 200 K
    _pair_peak_Ts = [
        float(T_grid[int(np.argmax(result.pair_log_likelihoods[p]))])
        for p in pyro.pairs
    ]
    loglik_xlim = (
        min(_pair_peak_Ts) - 200.0,
        max(_pair_peak_Ts) + 200.0,
    )
    loglik_ylim = (-3000.0, 100.0)

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    for row_idx, pair in enumerate(pyro.pairs):
        ci, cj = pair
        label = f"{ci}/{cj}"
        ratio_full = result.ratio_images[pair]
        ratio_roi = ratio_full[r0:r1, c0:c1]
        f_T = pyro.ratio_lut[pair]
        T_pair = pyro.temperatures_lut[pair]

        ax_img, ax_hist, ax_loglik = axes[row_idx]

        # --- Column 0: full-image ratio map with ROI rectangle ---
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

        # --- Column 1: ROI ratio histogram + LUT on twin y-axis ---
        roi_values = ratio_roi[np.isfinite(ratio_roi)].ravel()
        if roi_values.size:
            ax_hist.hist(roi_values, bins=HIST_BINS, color="steelblue", alpha=0.75)
        ax_hist.set_xlim(hist_xlim)
        ax_hist.set_ylim(hist_ylim)
        ax_hist.set_xlabel(f"{label} ratio")
        ax_hist.set_ylabel("Pixel count", color="steelblue")
        ax_hist.tick_params(axis="y", labelcolor="steelblue")
        # Shade the ratio range covered by the LUT to show which pixels are valid
        ax_hist.axvspan(float(f_T.min()), float(f_T.max()),
                        color="green", alpha=0.10, label="LUT range")
        # Twin y-axis: LUT curve mapping ratio -> temperature
        ax_lut = ax_hist.twinx()
        ax_lut.plot(f_T, T_pair, color="red", lw=1.5)
        ax_lut.set_ylim(T_axis_lim)
        ax_lut.set_ylabel("Temperature (K)", color="red")
        ax_lut.tick_params(axis="y", labelcolor="red")
        n_valid = result.n_valid_pixels_per_pair[pair]
        ax_hist.set_title(f"{label} ROI histogram + LUT  ({n_valid} valid px)")

        # --- Column 2: per-pair log-likelihood shifted to peak at 0 ---
        log_L = result.pair_log_likelihoods[pair]
        log_L_shifted = log_L - log_L.max()
        T_peak_pair = float(T_grid[int(np.argmax(log_L))])

        ax_loglik.plot(T_grid, log_L_shifted, color="darkorange", lw=1.5,
                       label=label)
        ax_loglik.axvline(T_peak_pair, color="darkorange", lw=1.0, alpha=0.6,
                          linestyle="--",
                          label=f"peak = {T_peak_pair:.1f} K")
        # Mark the joint MAP across all pairs for reference
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
        f"Bayesian ratio pyrometry  --  "
        f"MAP T = {result.map_temperature:.0f} K  --  "
        f"68% CI [{result.ci_68[0]:.0f}, {result.ci_68[1]:.0f}] K",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
