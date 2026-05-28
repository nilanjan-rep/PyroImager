"""
core.py – Bayesian multi-pair ratio pyrometer (isothermal ROI).

The library is intentionally agnostic of:
  * how quantum-efficiency (QE) data were obtained or stored — the caller
    supplies plain NumPy arrays.
  * image file format — images are loaded via pyrometer.image_io.load_image.

Physical constants come from scipy.constants (pure-Python, no compiled
extensions required).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import constants


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PyrometerResult:
    """
    Output of :meth:`RatioPyrometer.analyze`.

    The ROI is treated as **isothermal** at a single temperature ``T*``.
    The posterior ``π(T*)`` is computed on a discrete grid by sequentially
    multiplying the uniform prior on ``[T_min, T_max]`` by each pair's
    ROI-level likelihood.

    Attributes
    ----------
    temperatures_grid : ndarray, shape (N_T,)
        Temperature grid (K) on which the posterior is defined.
    posterior : ndarray, shape (N_T,)
        Final posterior π(T_k), normalized so ∫ π dT = 1.
    posterior_sequential : dict[int, ndarray]
        ``{0: π_0, 1: π_1, …, len(pairs): π_final}``, each normalized.
        ``π_0`` is the uniform prior.
    pair_log_likelihoods : dict[(ci, cj), ndarray]
        Per-pair ROI log-likelihood ``log L_ij(T_k)``, un-normalized.
    map_temperature : float
        Maximum-a-posteriori temperature (K).
    mean_temperature : float
        Posterior mean temperature (K).
    std_temperature : float
        Posterior standard deviation (K).
    ci_68, ci_95 : (float, float)
        Highest-density credible intervals (K).
    ratio_images : dict[(ci, cj), ndarray]
        Full-image WB-corrected ratio ``C_i/C_j``.  Pixels masked by
        ``min_signal_dn`` are NaN.
    n_valid_pixels_per_pair : dict[(ci, cj), int]
        Number of ROI pixels whose ratio fell inside the pair's LUT range
        and therefore contributed to that pair's likelihood.
    chi2_per_pair : dict[(ci, cj), float]
        χ² = Σ_p (R_p − f_ij(T_MAP))² / σ_R_p²  evaluated at the **joint
        MAP** temperature, summed over valid pixels for that pair.
    reduced_chi2_per_pair : dict[(ci, cj), float]
        χ² ÷ N_valid for each pair.  Close to 1 ⇒ the noise-weighted model
        residuals are consistent with the noise model.  ≫ 1 ⇒ either the
        noise model under-estimates the per-pixel uncertainty, or the
        isothermal-blackbody model is misspecified for that pair (the
        pair's individual best-fit T disagrees with the joint MAP).
    reduced_chi2_total : float
        Σ χ² over all pairs ÷ (Σ N_valid − 1), the global goodness-of-fit
        for the isothermal model fit.
    update_order : list of (ci, cj)
        Order of pairs used in the sequential update.
    """
    temperatures_grid: np.ndarray
    posterior: np.ndarray
    posterior_sequential: Dict[int, np.ndarray]
    pair_log_likelihoods: Dict[Tuple[str, str], np.ndarray]
    map_temperature: float
    mean_temperature: float
    std_temperature: float
    ci_68: Tuple[float, float]
    ci_95: Tuple[float, float]
    ratio_images: Dict[Tuple[str, str], np.ndarray]
    n_valid_pixels_per_pair: Dict[Tuple[str, str], int]
    chi2_per_pair: Dict[Tuple[str, str], float]
    reduced_chi2_per_pair: Dict[Tuple[str, str], float]
    reduced_chi2_total: float
    update_order: List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RatioPyrometer:
    """
    Bayesian multi-pair ratio pyrometer.

    Builds an ``S_i / S_j(T)`` lookup table for every channel pair that
    can be formed from the supplied QE data.  For RGB the pairs are
    ``(blue, red)``, ``(green, red)``, ``(green, blue)`` — taken in this
    order in the sequential posterior update.

    The :meth:`analyze` method assumes the ROI is **isothermal** and
    combines every (pixel, pair) measurement into a single posterior
    ``π(T*)`` on a temperature grid.

    Parameters
    ----------
    wavelengths_nm : array-like, shape (N,)
        Wavelength grid (nm), monotonically increasing.
    qe_channels : dict[str, array-like]
        ``{channel_name: QE array}``; each on *wavelengths_nm*.  Values in
        percent (>1) or fraction (≤1) — normalised to [0, 1] internally.
    t_range : (float, float), default (500.0, 3000.0)
        Temperature sweep extremes (K).
    n_temps : int, default 1000
        Grid resolution shared by the LUTs and the posterior.
    ir_filter : array-like or None, default None
        Optional filter transmission on *wavelengths_nm*.  ``None`` =
        full transmission.

    Attributes
    ----------
    pairs : list of (ci, cj)
        Pair list in canonical (update) order.
    ratio_lut : dict[(ci, cj), ndarray]
        ``f_ij(T_k)`` for each pair.
    temperatures_lut : dict[(ci, cj), ndarray]
        T grid (shared across pairs).
    is_monotonic : dict[(ci, cj), bool]
    monotonic_direction : dict[(ci, cj), {'increasing', 'decreasing', 'non-monotonic'}]
    """

    def __init__(
        self,
        wavelengths_nm: Union[np.ndarray, list, tuple],
        qe_channels: Dict[str, Union[np.ndarray, list, tuple]],
        t_range: Tuple[float, float] = (500.0, 3000.0),
        n_temps: int = 1000,
        ir_filter: Optional[Union[np.ndarray, list, tuple]] = None,
    ) -> None:
        if len(qe_channels) < 2:
            raise ValueError("Need at least 2 channels in qe_channels")

        self._wl_nm = np.asarray(wavelengths_nm, dtype=float)
        self._wl_m = self._wl_nm * 1e-9
        self.t_range = t_range
        self.n_temps = n_temps

        # IR filter
        if ir_filter is not None:
            filt = np.asarray(ir_filter, dtype=float)
            if filt.max() > 1.0:
                filt = filt / 100.0
        else:
            filt = np.ones_like(self._wl_nm)

        # Normalise each QE to [0, 1] and apply the filter
        self._qe_eff: Dict[str, np.ndarray] = {}
        for name, qe in qe_channels.items():
            q = np.asarray(qe, dtype=float)
            if q.max() > 1.0:
                q = q / 100.0
            self._qe_eff[name] = q * filt

        # Pre-compute integrated signal per channel on the T grid
        T, signals = self._compute_signals()
        self._temperatures_grid = T
        self._signals = signals

        # Canonical pair list (sets the update order)
        self.pairs = self._make_pair_list(list(qe_channels.keys()))

        # Build LUT for each pair + classify monotonicity
        self.ratio_lut: Dict[Tuple[str, str], np.ndarray] = {}
        self.temperatures_lut: Dict[Tuple[str, str], np.ndarray] = {}
        self.is_monotonic: Dict[Tuple[str, str], bool] = {}
        self.monotonic_direction: Dict[Tuple[str, str], str] = {}
        for ci, cj in self.pairs:
            self.ratio_lut[(ci, cj)] = signals[ci] / signals[cj]
            self.temperatures_lut[(ci, cj)] = T
            self._classify_monotonicity(ci, cj)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pair_list(channel_names: List[str]) -> List[Tuple[str, str]]:
        """
        Canonical update order.

        RGB:  (blue, red), (green, red), (green, blue)
              — monotonic, ambiguous, monotonic — so the non-monotonic
              pair is bracketed by disambiguating pairs.
        Other channel sets: alphabetical ``itertools.combinations``.
        """
        names_lc = {n.lower() for n in channel_names}
        if names_lc == {"red", "green", "blue"}:
            return [("blue", "red"), ("green", "red"), ("green", "blue")]
        return list(itertools.combinations(sorted(channel_names), 2))

    def _compute_signals(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Pre-compute S_i(T) = ∫ (λ/hc) B_λ(λ,T) Q_eff_i(λ) dλ for every channel."""
        h, c, k = constants.h, constants.c, constants.k
        T = np.linspace(self.t_range[0], self.t_range[1], self.n_temps)
        lam = self._wl_m

        # Planck spectral radiance vectorised over T  →  (n_temps, N_wl)
        x = (h * c) / (lam * k * T[:, np.newaxis])
        B = (2.0 * h * c**2 / lam**5) / np.expm1(x)
        BW = (lam / (h * c)) * B  # photon-flux-weighted Planck

        try:
            trapz = np.trapezoid           # NumPy ≥ 2.0
        except AttributeError:
            trapz = np.trapz               # NumPy < 2.0

        signals = {
            name: trapz(BW * qe_eff, lam, axis=1)
            for name, qe_eff in self._qe_eff.items()
        }
        return T, signals

    def _classify_monotonicity(self, ci: str, cj: str) -> None:
        R = self.ratio_lut[(ci, cj)]
        diffs = np.diff(R)
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n_total = len(diffs)
        if n_pos == n_total:
            self.is_monotonic[(ci, cj)] = True
            self.monotonic_direction[(ci, cj)] = "increasing"
        elif n_neg == n_total:
            self.is_monotonic[(ci, cj)] = True
            self.monotonic_direction[(ci, cj)] = "decreasing"
        else:
            self.is_monotonic[(ci, cj)] = False
            self.monotonic_direction[(ci, cj)] = "non-monotonic"
            n_viol = min(n_pos, n_neg)
            fraction = n_viol / n_total
            warnings.warn(
                f"LUT for {ci}/{cj} is non-monotonic "
                f"({n_viol}/{n_total} steps, {100.0 * fraction:.1f}%); "
                "the Bayesian fusion will resolve any ambiguity using the other pairs.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        roi: Optional[Tuple[int, int, int, int]] = None,
        channel_order: Tuple[str, ...] = ("red", "green", "blue"),
        wb_red: float = 1.0,
        wb_blue: float = 1.0,
        read_noise_dn: float = 0.86,
        gain: float = 2.72,
        max_dn: float = 255.0,
        min_signal_dn: float = 2.0,
    ) -> PyrometerResult:
        """
        Estimate ROI temperature by Bayesian multi-pair fusion (isothermal ROI).

        Parameters
        ----------
        image : str, Path, or ndarray
            Path to an image file (loaded via :func:`pyrometer.image_io.load_image`)
            or a pre-loaded ``(H, W, 3)`` array in normalised [0, 1].
        roi : (row0, row1, col0, col1) or None
            Pixel-index ROI.  ``None`` ⇒ full image.
        channel_order : tuple of str
            Names of the colour channels along axis 2.  Default
            ``('red', 'green', 'blue')`` matches PIL/imageio/PPM RGB order.
        wb_red, wb_blue : float, default 1.0
            White-balance gains relative to green (α_G ≡ 1).  Set these
            to the camera's R/G and B/G WB ratios so the library divides
            them out and recovers the true (sensor-electron) ratios.
        read_noise_dn : float, default 0.86
            Read noise in DN at the bit depth of the stored image.
            Default is the CS165CU 8-bit equivalent (13.75 DN-12bit / 16).
        gain : float, default 2.72
            Conversion gain in e⁻/DN (electrons per stored DN).  Default
            is CS165CU 8-bit equivalent (0.17 e⁻/DN-12bit × 16).
        max_dn : float, default 255
            Full-scale DN of the stored image (255 for 8-bit, 4095 for
            12-bit, 65535 for 16-bit).
        min_signal_dn : float, default 2.0
            Mask pixels where either channel is below this DN.

        Returns
        -------
        PyrometerResult
        """
        # ---- Load and convert image to DN ----
        if isinstance(image, (str, Path)):
            from .image_io import load_image
            img_norm = load_image(image)
        else:
            img_norm = np.asarray(image, dtype=float)
            if img_norm.ndim != 3 or img_norm.shape[2] < 3:
                raise ValueError(
                    f"Expected (H, W, 3) image array, got shape {img_norm.shape}"
                )

        img_dn = img_norm * max_dn
        H, W = img_dn.shape[:2]

        # ---- Channel-name → axis-2 index mapping ----
        ch_to_axis = {name.lower(): i for i, name in enumerate(channel_order)}
        for name in self._qe_eff:
            if name not in ch_to_axis:
                raise ValueError(
                    f"channel '{name}' not present in channel_order {channel_order}"
                )

        # ---- WB gains (green-referenced; unspecified channels default to 1.0) ----
        wb_gains: Dict[str, float] = {
            "red": float(wb_red),
            "green": 1.0,
            "blue": float(wb_blue),
        }
        for name in self._qe_eff:
            wb_gains.setdefault(name, 1.0)

        # ---- Per-channel DN images and per-pixel noise images ----
        C_dn: Dict[str, np.ndarray] = {}
        sigma_C: Dict[str, np.ndarray] = {}
        for name in self._qe_eff:
            ch = img_dn[:, :, ch_to_axis[name]]
            C_dn[name] = ch
            sigma_C[name] = np.sqrt(
                read_noise_dn**2 + np.maximum(ch, 0.0) / gain
            )

        # ---- ROI bounds ----
        if roi is None:
            r0, r1, c0, c1 = 0, H, 0, W
        else:
            r0, r1, c0, c1 = roi

        T_grid = self._temperatures_grid
        N_T = len(T_grid)

        # ---- Per-pair likelihood and bookkeeping ----
        ratio_images: Dict[Tuple[str, str], np.ndarray] = {}
        pair_log_L: Dict[Tuple[str, str], np.ndarray] = {}
        n_valid_per_pair: Dict[Tuple[str, str], int] = {}

        for ci, cj in self.pairs:
            alpha_ratio = wb_gains[cj] / wb_gains[ci]
            C_i, C_j = C_dn[ci], C_dn[cj]
            s_i, s_j = sigma_C[ci], sigma_C[cj]

            signal_mask = (C_i > min_signal_dn) & (C_j > min_signal_dn)

            # Full-image true ratio (WB-corrected), NaN where masked
            with np.errstate(divide="ignore", invalid="ignore"):
                R_full = alpha_ratio * (C_i / C_j)
            R_full = np.where(signal_mask, R_full, np.nan)
            ratio_images[(ci, cj)] = R_full

            # ROI extraction
            R_roi = R_full[r0:r1, c0:c1]
            C_i_roi = C_i[r0:r1, c0:c1]
            C_j_roi = C_j[r0:r1, c0:c1]
            s_i_roi = s_i[r0:r1, c0:c1]
            s_j_roi = s_j[r0:r1, c0:c1]

            # Ratio noise (relative noise is WB-invariant)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_i = s_i_roi / C_i_roi
                rel_j = s_j_roi / C_j_roi
            sigma_R = np.abs(R_roi) * np.sqrt(rel_i**2 + rel_j**2)

            # Validity: ratio in LUT range, noise finite & positive
            f_T = self.ratio_lut[(ci, cj)]
            r_min, r_max = float(f_T.min()), float(f_T.max())
            valid = (
                np.isfinite(R_roi)
                & (R_roi >= r_min)
                & (R_roi <= r_max)
                & np.isfinite(sigma_R)
                & (sigma_R > 0)
            )
            n_valid_per_pair[(ci, cj)] = int(valid.sum())

            R_v = R_roi[valid]
            sigma_v = sigma_R[valid]

            if R_v.size == 0:
                # No ROI pixel constrains this pair → flat (zero) log-L
                pair_log_L[(ci, cj)] = np.zeros(N_T)
                continue

            # logL(T_k) = -½ · Σ_p (R_p - f(T_k))² / σ_p²
            #          = -½ · ( S2 - 2 f(T_k) S1 + f(T_k)² S0 )
            # with  S_n = Σ_p R_p^n / σ_p².  Avoids forming (N_T × N_pix) matrix.
            inv_sigma2 = 1.0 / (sigma_v ** 2)
            S0 = float(np.sum(inv_sigma2))
            S1 = float(np.sum(R_v * inv_sigma2))
            S2 = float(np.sum(R_v ** 2 * inv_sigma2))
            pair_log_L[(ci, cj)] = -0.5 * (S2 - 2.0 * f_T * S1 + (f_T ** 2) * S0)

        # ---- Sequential posterior update ----
        log_pi: Dict[int, np.ndarray] = {0: np.zeros(N_T)}  # uniform prior (constant absorbed in normalisation)
        for step, pair in enumerate(self.pairs, start=1):
            log_pi[step] = log_pi[step - 1] + pair_log_L[pair]

        posterior_sequential: Dict[int, np.ndarray] = {
            step: self._normalize_log_posterior(lp, T_grid)
            for step, lp in log_pi.items()
        }
        final_posterior = posterior_sequential[len(self.pairs)]

        # ---- Summary statistics on the final posterior ----
        try:
            trapz = np.trapezoid
        except AttributeError:
            trapz = np.trapz

        map_idx = int(np.argmax(final_posterior))
        map_T = float(T_grid[map_idx])
        mean_T = float(trapz(T_grid * final_posterior, T_grid))
        var_T = float(trapz((T_grid - mean_T) ** 2 * final_posterior, T_grid))
        std_T = float(np.sqrt(max(var_T, 0.0)))
        ci_68 = self._hdi(final_posterior, T_grid, 0.68)
        ci_95 = self._hdi(final_posterior, T_grid, 0.95)

        # ---- Goodness-of-fit at the joint MAP ----
        # The stored log-L for each pair is the bare quadratic
        #     log L_ij(T) = -½ Σ_p (R_p − f_ij(T))² / σ_p²
        # so χ² = −2 · log L at the joint MAP temperature.
        chi2_per_pair: Dict[Tuple[str, str], float] = {}
        reduced_chi2_per_pair: Dict[Tuple[str, str], float] = {}
        total_chi2 = 0.0
        total_n = 0
        for pair in self.pairs:
            n_valid = n_valid_per_pair[pair]
            if n_valid == 0:
                chi2_per_pair[pair] = float("nan")
                reduced_chi2_per_pair[pair] = float("nan")
                continue
            chi2 = -2.0 * float(pair_log_L[pair][map_idx])
            chi2_per_pair[pair] = chi2
            reduced_chi2_per_pair[pair] = chi2 / n_valid
            total_chi2 += chi2
            total_n += n_valid

        # one fitted parameter (T*) across all pairs
        dof_total = max(total_n - 1, 1)
        reduced_chi2_total = float(total_chi2 / dof_total) if total_n > 0 else float("nan")

        return PyrometerResult(
            temperatures_grid=T_grid,
            posterior=final_posterior,
            posterior_sequential=posterior_sequential,
            pair_log_likelihoods=pair_log_L,
            map_temperature=map_T,
            mean_temperature=mean_T,
            std_temperature=std_T,
            ci_68=ci_68,
            ci_95=ci_95,
            ratio_images=ratio_images,
            n_valid_pixels_per_pair=n_valid_per_pair,
            chi2_per_pair=chi2_per_pair,
            reduced_chi2_per_pair=reduced_chi2_per_pair,
            reduced_chi2_total=reduced_chi2_total,
            update_order=list(self.pairs),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_log_posterior(
        log_p: np.ndarray, T_grid: np.ndarray
    ) -> np.ndarray:
        """Convert a log-posterior to a normalised probability density via log-sum-exp + trapezoid."""
        try:
            trapz = np.trapezoid
        except AttributeError:
            trapz = np.trapz
        # Shift for numerical stability
        lp = log_p - log_p.max()
        p = np.exp(lp)
        integral = float(trapz(p, T_grid))
        if integral <= 0 or not np.isfinite(integral):
            return np.ones_like(T_grid) / (T_grid[-1] - T_grid[0])
        return p / integral

    @staticmethod
    def _hdi(
        posterior: np.ndarray,
        T_grid: np.ndarray,
        mass: float,
    ) -> Tuple[float, float]:
        """Highest-density credible interval covering *mass* of the posterior mass."""
        if posterior.sum() == 0:
            return (float(T_grid[0]), float(T_grid[-1]))
        dT = float(T_grid[1] - T_grid[0])
        order = np.argsort(-posterior)                  # densest first
        sorted_p = posterior[order]
        cum_mass = np.cumsum(sorted_p * dT)
        if cum_mass[-1] > 0:
            cum_mass = cum_mass / cum_mass[-1]
        idx = int(np.searchsorted(cum_mass, mass, side="right"))
        if idx >= len(cum_mass):
            idx = len(cum_mass) - 1
        included = order[: idx + 1]
        return (float(T_grid[included].min()), float(T_grid[included].max()))

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def plot_lut(self, pair: Tuple[str, str], ax=None):
        """
        Plot a single pair's S_i/S_j(T) LUT.  Non-monotone steps are marked
        in red.

        Parameters
        ----------
        pair : (str, str)
            Channel pair, e.g. ``("blue", "red")``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created if None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ci, cj = pair
        T = self.temperatures_lut[pair]
        R = self.ratio_lut[pair]
        ax.plot(T, R, color="steelblue", lw=1.5)
        if not self.is_monotonic[pair]:
            diffs = np.diff(R)
            dominant = np.sign(np.median(diffs))
            bad = np.where(np.sign(diffs) != dominant)[0]
            ax.scatter(
                T[bad], R[bad], color="red", zorder=5, s=10,
                label="non-monotone step",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(rf"$S_{{\mathrm{{{ci}}}}} / S_{{\mathrm{{{cj}}}}}$")
        ax.set_title(f"{ci}/{cj}  ({self.monotonic_direction[pair]})")
        ax.grid(True, alpha=0.3)
        return ax
