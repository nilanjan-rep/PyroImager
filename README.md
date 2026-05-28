# PyroImager

A Python library for estimating temperature from a region-of-interest (ROI) in a colour image using ratio pyrometry. The algorithm assumes that the colour signal inside the ROI comes solely from black-body radiation, and uses Bayesian inference to combine measurements from all available channel pairs into a single posterior distribution on temperature.

## AI Usage Disclosure
This project was co-authored and developed with the assistance of *Claude code*. All code, documentation, and logic have been reviewed, tested, and curated by a human developer to ensure quality and accuracy.

## Theory

### Black-body Radiation and Camera Signals

The spectral radiance $B_\lambda(\lambda,T)$ (W sr$^{-1}$ m$^{-3}$) of a black body at temperature $T$ is given by Planck's law:

$$B_\lambda(\lambda,T) = \frac{2hc^2}{\lambda^5} \frac{1}{\exp\!\left(\dfrac{hc}{\lambda k_B T}\right)-1}$$

where $h$, $c$, and $k_B$ are the Planck constant, speed of light, and Boltzmann constant.
For a pixel exposed to black-body radiation with emissivity $\epsilon$, the signal $S_i$ (ADC units) in colour channel $i$ is:

$$S_i = \epsilon \,\Omega\, A\, \tau\, G \int_0^\infty \frac{\lambda}{hc}\, B_\lambda(\lambda,T)\, Q_i(\lambda)\,\mathrm{d}\lambda$$

where $A$ and $\Omega$ are the pixel area and subtended solid angle, $\tau$ is the exposure time, $G$ is the conversion gain (ADC counts per electron), and $Q_i(\lambda)$ is the quantum efficiency of channel $i$ as a function of wavelength.

### Ratio Pyrometry

Taking the ratio of two channel signals cancels the unknown factors $\epsilon$, $\Omega$, $A$, $\tau$, and $G$:

$$f_{ij}(T) = \frac{S_i}{S_j} = \frac{\displaystyle\int \frac{\lambda}{hc}\, B_\lambda(\lambda,T)\, Q_i(\lambda)\,\mathrm{d}\lambda}{\displaystyle\int \frac{\lambda}{hc}\, B_\lambda(\lambda,T)\, Q_j(\lambda)\,\mathrm{d}\lambda}$$

Given a measured ratio $R_{ij}$, the temperature is found by inverting $f_{ij}(T)$.

### Bayesian Inference (Isothermal ROI)

The library models the entire ROI as isothermal at a single temperature $T^*$. For each channel pair $(i, j)$ at each ROI pixel $p$:

**1. Noise model.** Read noise and Poisson shot noise in channel $i$:
$$\sigma_{C_i}^{(p)} = \sqrt{\sigma_\mathrm{read}^2 + C_i^{(p)}/g}$$

The ratio noise follows by error propagation (the relative noise is white-balance-invariant):
$$\frac{\sigma_R^{(p)}}{R^{(p)}} = \sqrt{\!\left(\frac{\sigma_{C_i}}{C_i}\right)^{\!2} + \left(\frac{\sigma_{C_j}}{C_j}\right)^{\!2}}$$

**2. Per-pair log-likelihood.** On a discrete temperature grid $\{T_k\}$:
$$\log \mathcal{L}_{ij}(T_k) = -\frac{1}{2}\sum_p \frac{\bigl(R_{ij}^{(p)} - f_{ij}(T_k)\bigr)^2}{\sigma_{R,p}^2}$$

Pixels whose ratio falls outside the LUT range of $f_{ij}$ contribute no information to that pair but are still used by the other pairs.

**3. Sequential Bayesian update.** Starting from a uniform prior $\pi_0$ on $[T_\mathrm{min}, T_\mathrm{max}]$:
$$\log \pi_n(T) = \log \pi_{n-1}(T) + \log \mathcal{L}_{\mathrm{pair}_n}(T)$$

Each $\pi_n$ is normalized via the trapezoid rule. For RGB sensors the update order is (blue/red) -> (green/red) -> (green/blue): the two monotonic pairs bracket the potentially non-monotonic green/red pair, resolving any ambiguity.

**4. Summary statistics.** MAP, posterior mean and standard deviation, and 68% / 95% highest-density credible intervals are computed from the final posterior $\pi_3$.

**5. Goodness of fit.** At the joint MAP temperature $T^*_\mathrm{MAP}$:
$$\chi^2_{ij} = -2\log\mathcal{L}_{ij}(T^*_\mathrm{MAP}),\qquad \chi^2_{ij}/N_\mathrm{valid} \approx 1 \Rightarrow \text{model fits the noise}$$

## Installation

Clone the repository and install with pip:

```bash
git clone https://github.com/nilanjan-rep/PyroImager.git
cd ImagePyrometer
pip install -e .
```

Dependencies:

```
numpy
scipy
matplotlib
Pillow or imageio    # for image loading (.ppm files work without either)
openpyxl             # example only, to read the xlsx QE file
```

## Camera Parameters (CS165CU/Blackfly S BFS-PGE-16S2C)

The example targets the **Thorlabs CS165CU**/**FLIR Blackfly S BFS-PGE-16S2C** or equivalent cameras using the **Sony IMX273** sensor (1440 x 1080) with the IR filter physically removed. Removing the filter gives the blue pixels sensitivity into the near-IR, widening the effective spectral separation between channels and improving temperature sensitivity.

The camera has a 12-bit ADC, but the test image stores only the top 8 MSBs, so 1 stored DN corresponds to 16 native DN:

| Parameter | 12-bit (native ADC) | 8-bit (stored image) |
|---|---|---|
| Read noise | 13.75 DN | **0.86 DN** |
| Conversion gain | 0.17 e$^-$/DN | **2.72 e$^-$/DN** |
| Full scale | 4095 DN | **255 DN** |

The test image was captured with automatic white balance disabled (`wb_red = wb_blue = 1`). When white balance is enabled, set `wb_red` and `wb_blue` to the camera's R/G and B/G gain ratios; the library divides them out to recover the true sensor-electron ratios.

For a different camera, override `read_noise_dn`, `gain`, and `max_dn` in the `analyze()` call.

## Usage

```python
import numpy as np
from pyrometer import RatioPyrometer

# Supply QE arrays (percent or fraction) on a wavelength grid (nm)
wl_nm  = ...   # 1-D array, e.g. 400-1000 nm in 5 nm steps
qe_red = ...   # same length as wl_nm
qe_grn = ...
qe_blu = ...

pyro = RatioPyrometer(
    wavelengths_nm=wl_nm,
    qe_channels={"red": qe_red, "green": qe_grn, "blue": qe_blu},
    t_range=(500.0, 2500.0),   # temperature grid range (K)
    n_temps=1000,
    # ir_filter=None  <-- default: full transmission
)

result = pyro.analyze(
    image="path/to/image.ppm",      # or an (H, W, 3) float64 array in [0, 1]
    roi=(r0, r1, c0, c1),           # pixel-index ROI; None = full image
    channel_order=("red", "green", "blue"),
    wb_red=1.0,                     # R/G white-balance gain (1.0 if WB off)
    wb_blue=1.0,                    # B/G white-balance gain
    read_noise_dn=0.86,             # CS165CU 8-bit defaults
    gain=2.72,
    max_dn=255,
)

print(f"MAP temperature: {result.map_temperature:.1f} K")
print(f"68% CI: [{result.ci_68[0]:.1f}, {result.ci_68[1]:.1f}] K")
```

### Generating the analysis figure

```python
from pyrometer.plotting import plot_analysis

fig = plot_analysis(result, pyro, "path/to/image.ppm", roi=(r0, r1, c0, c1))
fig.savefig("pyrometry_result.png", dpi=150)
```

### Result fields

| Field | Description |
|---|---|
| `temperatures_grid` | 1-D temperature grid (K) |
| `posterior` | Final posterior $\pi_3(T_k)$, normalized so $\int\pi\,\mathrm{d}T = 1$ |
| `posterior_sequential` | `{0: pi_0, ..., 3: pi_3}`, each normalized |
| `pair_log_likelihoods` | `{pair: log L_ij(T_k)}` for each pair |
| `map_temperature` | MAP temperature (K) |
| `mean_temperature` | Posterior mean (K) |
| `std_temperature` | Posterior standard deviation (K) |
| `ci_68`, `ci_95` | Highest-density credible intervals (K) |
| `ratio_images` | `{pair: 2-D WB-corrected ratio map}` over the full image |
| `n_valid_pixels_per_pair` | Number of ROI pixels within LUT range for each pair |
| `chi2_per_pair` | $\chi^2$ at the joint MAP for each pair |
| `reduced_chi2_per_pair` | $\chi^2 / N_\mathrm{valid}$; close to 1 means a good fit |
| `reduced_chi2_total` | $\sum\chi^2 / (\sum N - 1)$, global goodness of fit |

## Running the Example

```bash
cd example/
python example_cs165cu.py
```

This produces three figures and a console summary:

- `pyrometry_result.png` -- 3 x 3 analysis grid (ratio images, histograms, log-likelihoods)
- `final_posterior.png` -- final log-posterior $\log\pi_3(T)$ with MAP and 68% CI
- `roi_ratio_images.png` -- ROI-only ratio maps (auto-scaled per pair)

Edit the `ROI` tuple in `example_cs165cu.py` to select the hot region in your image.

## Repository Layout

```
pyrometer/
    __init__.py      # exports RatioPyrometer, PyrometerResult, plot_analysis
    core.py          # RatioPyrometer class + PyrometerResult dataclass
    image_io.py      # format-agnostic image loader (PIL -> imageio -> built-in PPM)
    plotting.py      # plot_analysis() -- standard 3 x 3 analysis figure
example/
    example_cs165cu.py                 # end-to-end usage script
    cs165cu_quantum_efficiency.xlsx    # CS165CU sensor QE data
    test.ppm                           # 1440 x 1080 PPM test image (WB disabled)
```
