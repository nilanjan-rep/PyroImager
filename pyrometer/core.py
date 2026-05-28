"""
core.py -- Bayesian multi-pair ratio pyrometer (isothermal ROI).

The library is intentionally agnostic of:
  * how quantum-efficiency (QE) data were obtained or stored -- the caller
    supplies plain NumPy arrays.
  * image file format -- images are loaded via pyrometer.image_io.load_image.

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

    The ROI is treated as isothermal at a single temperature T*.
    The posterior pi(T*) is computed on a discrete grid by sequentially
    multiplying the uniform prior on [T_min, T_max] by each pair's
    ROI-level likelihood.

    Attributes
    ----------
    temperatures_grid : ndarray, shape (N_T,)
        Temperature grid (K) on which the posterior is defined.
    posterior : ndarray, shape (N_T,)
        Final posterior pi(T_k), normalized so integral(pi dT) = 1.
    posterior_sequential : dict[int, ndarray]
        {0: pi_0, 1: pi_1, ..., len(pairs): pi_final}, each normalized.
        pi_0 is the uniform prior.
    pair_log_likelihoods : dict[(ci, cj), ndarray]
        Per-pair ROI log-likelihood log_L_ij(T_k), un-normalized.
    map_temperature : float
        Maximum-a-posteriori temperature (K).
    mean_temperature : float
        Posterior mean temperature (K).
    std_temperature : float
        Posterior standard deviation (K).
    ci_68, ci_95 : (float, float)
        Highest-density credible intervals (K).
    ratio_images : dict[(ci, cj), ndarray]
        Full-image WB-corrected ratio C_i/C_j.  Pixels masked by
        min_signal_dn are NaN.
    n_valid_pixels_per_pair : dict[(ci, cj), int]
        Number of ROI pixels whose ratio fell inside the pair's LUT range
        and therefore contributed to that pair's likelihood.
    chi2_per_pair : dict[(ci, cj), float]
        chi^2 = sum_p (R_p - f_ij(T_MAP))^2 / sigma_p^2 evaluated at the
        joint MAP temperature, summed over valid pixels for that pair.
    reduced_chi2_per_pair : dict[(ci, cj), float]
        chi^2 / N_valid for each pair.  Close to 1 => the noise-weighted
        model residuals are consistent with the noise model.  >> 1 =>
        either the noise model under-estimates the per-pixel uncertainty,
        or the isothermal-blackbody model is misspecified for that pair
        (the pair's individual best-fit T disagrees with the joint MAP).
    reduced_chi2_total : float
        sum(chi^2) over all pairs / (sum(N_valid) - 1), the global
        goodness-of-fit for the isothermal model fit.
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

    Builds an S_i / S_j(T) lookup table for every channel pair that
    can be formed from the supplied QE data.  For RGB the pairs are
    (blue, red), (green, red), (green, blue) -- taken in this
    order in the sequential posterior update.

    The :meth:`analyze` method assumes the ROI is isothermal and
    combines every (pixel, pair) measurement into a single posterior
    pi(T*) on a temperature grid.

    Parameters
    ----------
    wavelengths_nm : array-like, shape (N,)
        Wavelength grid (nm), strictly increasing.
    qe_channels : dict[str, array-like]
        {channel_name: QE array}; each sampled on wavelengths_nm.  Values in
        percent (>1) or fraction (<=1) -- normalized to [0, 1] internally.
    t_range : (float, float), default (500.0, 3000.0)
        Temperature sweep extremes (K).
    n_temps : int, default 1000
        Grid resolution shared by the LUTs and the posterior.
    ir_filter : array-like or None, default None
        Optional filter transmission on wavelengths_nm.  None means
        full transmission (unity across all wavelengths).

    Attributes
    ----------
    pairs : list of (ci, cj)
        Pair list in canonical (update) order.
    ratio_lut : dict[(ci, cj), ndarray]
        f_ij(T_k) for each pair.
    temperatures_lut : dict[(ci, cj), ndarray]
        T grid (shared across pairs).
    is_monotonic : dict[(ci, cj), bool]
    monotonic_direction : dict[(ci, cj), str]
        One of 'increasing', 'decreasing', or 'non-monotonic'.
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
            raise ValueError(
                f"Need at least 2 channels in qe_channels; got {len(qe_channels)}"
            )

        wl = np.asarray(wavelengths_nm, dtype=float)
        if wl.ndim != 1:
            raise ValueError(
                f"wavelengths_nm must be a 1-D array; got shape {wl.shape}"
            )
        if len(wl) < 2:
            raise ValueError("wavelengths_nm must have at least 2 points")
        if not np.all(np.diff(wl) > 0):
            raise ValueError("wavelengths_nm must be strictly increasing")
        self._wl_nm = wl
        self._wl_m = wl * 1e-9

        t_range = (float(t_range[0]), float(t_range[1]))
        if t_range[0] >= t_range[1]:
            raise ValueError(
                f"t_range must satisfy T_min < T_max; got {t_range}"
            )
        if t_range[0] <= 0:
            raise ValueError(
                f"t_range[0] must be a positive absolute temperature (K); "
                f"got {t_range[0]}"
            )
        self.t_range = t_range

        n_temps = int(n_temps)
        if n_temps < 2:
            raise ValueError(f"n_temps must be >= 2; got {n_temps}")
        self.n_temps = n_temps

        # IR filter: None means full transmission across all wavelengths
        if ir_filter is not None:
            filt = np.asarray(ir_filter, dtype=float)
            if filt.ndim != 1 or len(filt) != len(self._wl_nm):
                raise ValueError(
                    f"ir_filter must be a 1-D array of length {len(self._wl_nm)}; "
                    f"got shape {filt.shape}"
                )
            if not np.all(np.isfinite(filt)):
                raise ValueError("ir_filter contains non-finite values")
            if np.any(filt < 0):
                raise ValueError("ir_filter contains negative values")
            if filt.max() > 1.0:
                filt = filt / 100.0
        else:
            filt = np.ones(len(self._wl_nm))

        # Normalize each QE curve to [0, 1] and apply the filter
        self._qe_eff: Dict[str, np.ndarray] = {}
        for name, qe in qe_channels.items():
            q = np.asarray(qe, dtype=float)
            if q.ndim != 1 or len(q) != len(self._wl_nm):
                raise ValueError(
                    f"QE array for channel '{name}' must be 1-D with length "
                    f"{len(self._wl_nm)}; got shape {q.shape}"
                )
            if not np.all(np.isfinite(q)):
                raise ValueError(
                    f"QE array for channel '{name}' contains non-finite values"
                )
            if np.any(q < 0):
                raise ValueError(
                    f"QE array for channel '{name}' contains negative values"
                )
            if q.max() > 1.0:
                q = q / 100.0
            self._qe_eff[name] = q * filt

        # Pre-compute integrated signal per channel on the T grid
        T, signals = self._compute_signals()
        self._temperatures_grid = T
        self._signals = signals

        # Canonical pair list sets the Bayesian update order
        self.pairs = self._make_pair_list(list(qe_channels.keys()))

        # Build LUT for each pair and classify monotonicity
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
        Return pairs in canonical update order.

        RGB:  (blue, red), (green, red), (green, blue)
              The ordering is monotonic -> ambiguous -> monotonic so that
              the non-monotonic green/red pair is bracketed by pairs that
              sharply localize the temperature, resolving any ambiguity.
        Other channel sets: alphabetical itertools.combinations.
        """
        names_lc = {n.lower() for n in channel_names}
        if names_lc == {"red", "green", "blue"}:
            return [("blue", "red"), ("green", "red"), ("green", "blue")]
        return list(itertools.combinations(sorted(channel_names), 2))

    def _compute_signals(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Pre-compute S_i(T) for every channel over the temperature grid.

        S_i(T) = integral (lam/hc) * B_lam(lam, T) * Q_eff_i(lam) dlam

        The (lam/hc) factor converts spectral radiance B_lam (W sr^-1 m^-3)
        to photon flux, making S_i proportional to the number of photo-
        electrons generated per unit time -- matching what the ADC records.
        """
        h, c, k = constants.h, constants.c, constants.k
        T = np.linspace(self.t_range[0], self.t_range[1], self.n_temps)
        lam = self._wl_m

        # Planck spectral radiance vectorized over T: shape (n_temps, N_wl)
        x = (h * c) / (lam * k * T[:, np.newaxis])
        B = (2.0 * h * c**2 / lam**5) / np.expm1(x)
        # Convert energy radiance to photon-flux-weighted radiance
        BW = (lam / (h * c)) * B

        try:
            trapz = np.trapezoid   # NumPy >= 2.0
        except AttributeError:
            trapz = np.trapz       # NumPy < 2.0

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
            Path to an image file (loaded via pyrometer.image_io.load_image)
            or a pre-loaded (H, W, 3) array normalized to [0, 1].
        roi : (row0, row1, col0, col1) or None
            Pixel-index ROI (half-open: row0 <= r < row1, col0 <= c < col1).
            None means the full image.
        channel_order : tuple of str
            Names of the colour channels along axis 2.  Default
            ('red', 'green', 'blue') matches PIL/imageio/PPM RGB order.
        wb_red, wb_blue : float, default 1.0
            White-balance gains relative to green (alpha_G = 1).  Set these
            to the camera's R/G and B/G WB ratios so the library divides
            them out and recovers the true (sensor-electron) ratios.
        read_noise_dn : float, default 0.86
            Read noise in DN at the bit depth of the stored image.
            Default is the CS165CU 8-bit equivalent (13.75 DN-12bit / 16).
        gain : float, default 2.72
            Conversion gain in e-/DN (electrons per stored DN).  Default
            is the CS165CU 8-bit equivalent (0.17 e-/DN-12bit * 16).
        max_dn : float, default 255
            Full-scale DN of the stored image (255 for 8-bit, 4095 for
            12-bit, 65535 for 16-bit).
        min_signal_dn : float, default 2.0
            Mask pixels where either channel is below this DN to suppress
            noise-dominated ratios at the detector floor.

        Returns
        -------
        PyrometerResult
        """
        # Validate scalar camera parameters up front
        if gain <= 0:
            raise ValueError(f"gain must be positive; got {gain}")
        if max_dn <= 0:
            raise ValueError(f"max_dn must be positive; got {max_dn}")
        if read_noise_dn < 0:
            raise ValueError(f"read_noise_dn must be non-negative; got {read_noise_dn}")
        if min_signal_dn < 0:
            raise ValueError(f"min_signal_dn must be non-negative; got {min_signal_dn}")

        # Load and convert image to DN
        if isinstance(image, (str, Path)):
            from .image_io import load_image
            img_norm = load_image(image)
        else:
            img_norm = np.asarray(image, dtype=float)
            if img_norm.ndim != 3 or img_norm.shape[2] < 3:
                raise ValueError(
                    f"Expected (H, W, 3) image array; got shape {img_norm.shape}"
                )

        img_dn = img_norm * max_dn
        H, W = img_dn.shape[:2]

        # Validate channel_order covers every channel in qe_channels
        if len(channel_order) < img_dn.shape[2]:
            raise ValueError(
                f"channel_order has {len(channel_order)} entries but image has "
                f"{img_dn.shape[2]} channels"
            )
        ch_to_axis = {name.lower(): i for i, name in enumerate(channel_order)}
        for name in self._qe_eff:
            if name not in ch_to_axis:
                raise ValueError(
                    f"Channel '{name}' from qe_channels not found in "
                    f"channel_order {channel_order}"
                )

        # Validate and resolve ROI bounds
        if roi is not None:
            r0, r1, c0, c1 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            if not (0 <= r0 < r1 <= H):
                raise ValueError(
                    f"roi rows ({r0}, {r1}) out of bounds for image height {H}; "
                    "require 0 <= row0 < row1 <= H"
                )
            if not (0 <= c0 < c1 <= W):
                raise ValueError(
                    f"roi cols ({c0}, {c1}) out of bounds for image width {W}; "
                    "require 0 <= col0 < col1 <= W"
                )
        else:
            r0, r1, c0, c1 = 0, H, 0, W

        # Green is the WB reference channel (gain = 1); others are user-supplied
        wb_gains: Dict[str, float] = {
            "red": float(wb_red),
            "green": 1.0,
            "blue": float(wb_blue),
        }
        for name in self._qe_eff:
            wb_gains.setdefault(name, 1.0)

        # Per-channel DN images and per-pixel noise (read + Poisson shot)
        C_dn: Dict[str, np.ndarray] = {}
        sigma_C: Dict[str, np.ndarray] = {}
        for name in self._qe_eff:
            ch = img_dn[:, :, ch_to_axis[name]]
            C_dn[name] = ch
            # Clamp negative DN to 0 before computing the shot-noise term
            sigma_C[name] = np.sqrt(read_noise_dn**2 + np.maximum(ch, 0.0) / gain)

        T_grid = self._temperatures_grid
        N_T = len(T_grid)

        # Per-pair bookkeeping
        ratio_images: Dict[Tuple[str, str], np.ndarray] = {}
        pair_log_L: Dict[Tuple[str, str], np.ndarray] = {}
        n_valid_per_pair: Dict[Tuple[str, str], int] = {}

        for ci, cj in self.pairs:
            # WB correction factor: multiply the raw ratio by alpha_j/alpha_i to
            # recover the true (sensor-electron) ratio from the WB-scaled image.
            # This is WB-invariant in the sense that sigma_R/R does not depend on WB.
            alpha_ratio = wb_gains[cj] / wb_gains[ci]
            C_i, C_j = C_dn[ci], C_dn[cj]
            s_i, s_j = sigma_C[ci], sigma_C[cj]

            signal_mask = (C_i > min_signal_dn) & (C_j > min_signal_dn)

            # Full-image WB-corrected ratio; NaN where signal is too low
            with np.errstate(divide="ignore", invalid="ignore"):
                R_full = alpha_ratio * (C_i / C_j)
            R_full = np.where(signal_mask, R_full, np.nan)
            ratio_images[(ci, cj)] = R_full

            # Extract ROI pixels
            R_roi = R_full[r0:r1, c0:c1]
            C_i_roi = C_i[r0:r1, c0:c1]
            C_j_roi = C_j[r0:r1, c0:c1]
            s_i_roi = s_i[r0:r1, c0:c1]
            s_j_roi = s_j[r0:r1, c0:c1]

            # Ratio noise by error propagation:
            #   sigma_R / R = sqrt((sigma_Ci/Ci)^2 + (sigma_Cj/Cj)^2)
            # The relative noise is WB-invariant, so we use the raw DN sigma.
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_i = s_i_roi / C_i_roi
                rel_j = s_j_roi / C_j_roi
            sigma_R = np.abs(R_roi) * np.sqrt(rel_i**2 + rel_j**2)

            # A pixel is valid if its ratio is within the LUT range and noise is finite
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
                # No valid pixels for this pair; flat log-L contributes nothing
                pair_log_L[(ci, cj)] = np.zeros(N_T)
                continue

            # Vectorized Gaussian log-likelihood over the T grid:
            #   log_L(T_k) = -0.5 * sum_p (R_p - f(T_k))^2 / sigma_p^2
            #              = -0.5 * (S2 - 2*f(T_k)*S1 + f(T_k)^2 * S0)
            # where S_n = sum_p R_p^n / sigma_p^2.
            # Pre-computing S0, S1, S2 avoids the (N_T x N_pix) outer product.
            inv_sigma2 = 1.0 / (sigma_v ** 2)
            S0 = float(np.sum(inv_sigma2))
            S1 = float(np.sum(R_v * inv_sigma2))
            S2 = float(np.sum(R_v ** 2 * inv_sigma2))
            pair_log_L[(ci, cj)] = -0.5 * (S2 - 2.0 * f_T * S1 + (f_T ** 2) * S0)

        # Sequential posterior update:
        #   log pi_n(T) = log pi_{n-1}(T) + log_L_{pair_n}(T)
        # Start from a uniform prior (constant log-pi = 0, absorbed by normalization)
        log_pi: Dict[int, np.ndarray] = {0: np.zeros(N_T)}
        for step, pair in enumerate(self.pairs, start=1):
            log_pi[step] = log_pi[step - 1] + pair_log_L[pair]

        posterior_sequential: Dict[int, np.ndarray] = {
            step: self._normalize_log_posterior(lp, T_grid)
            for step, lp in log_pi.items()
        }
        final_posterior = posterior_sequential[len(self.pairs)]

        # Summary statistics on the final posterior
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

        # Goodness-of-fit at the joint MAP temperature.
        # The stored log-L is the bare quadratic
        #   log_L_ij(T) = -0.5 * sum_p (R_p - f_ij(T))^2 / sigma_p^2
        # so chi^2 = -2 * log_L at the MAP.
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

        # One fitted parameter (T*) is shared across all pairs
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
        """
        Convert a log-posterior to a normalized probability density.

        Subtracts the maximum before exponentiation (log-sum-exp trick) to
        prevent overflow, then normalizes via the trapezoid rule so that
        integral(pi dT) = 1.
        """
        try:
            trapz = np.trapezoid
        except AttributeError:
            trapz = np.trapz
        lp = log_p - log_p.max()
        p = np.exp(lp)
        integral = float(trapz(p, T_grid))
        if integral <= 0 or not np.isfinite(integral):
            # Degenerate case: return a uniform density
            return np.ones_like(T_grid) / (T_grid[-1] - T_grid[0])
        return p / integral

    @staticmethod
    def _hdi(
        posterior: np.ndarray,
        T_grid: np.ndarray,
        mass: float,
    ) -> Tuple[float, float]:
        """
        Highest-density credible interval (HDI) enclosing `mass` of the posterior.

        Grid points are included in order of decreasing density until the
        enclosed probability mass reaches `mass`.  The HDI is then the
        smallest contiguous temperature range covering those points.
        """
        if posterior.sum() == 0:
            return (float(T_grid[0]), float(T_grid[-1]))
        dT = float(T_grid[1] - T_grid[0])
        order = np.argsort(-posterior)   # densest grid points first
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
            Channel pair, e.g. ("blue", "red").
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created if None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if pair not in self.ratio_lut:
            raise ValueError(
                f"Pair {pair} not found. Available pairs: {list(self.ratio_lut.keys())}"
            )
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
