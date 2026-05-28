# PyroImager
This is a small image processing library written in pure python which extracts temperature from a region-of-interest (ROI) using ratio pyrometery. The processing algorithm assumes that color information inside the ROI is solely due to black-body radiation emitted from the object inside the ROI.

## Theory
The spectral radiance $B_\lambda(\lambda,T)$ (in W/sr/m^3) at wavelength $\lambda$ of an object at temperature $T$, which is emitting black-body radiation is given by Planck's law,

$$ B_\lambda(\lambda,T) = \frac{2hc^2}{\lambda^5} \frac{1}{\exp(\frac{hc}{\lambda k_B T})-1}\,,$$

where $h$, $c$ and $k_B$ are the Planck constant, speed of light, and the Boltzmann constant. Assuming that light from an object with emissivity $\epsilon$ is incident on a single pixel in the ROI, the signals $S_i$ (in ADC units) from the red, green, and blue channels are given by,

$$ S_i = \epsilon \Omega A \tau G \int_0^\infty \frac{\lambda}{hc} B_\lambda(\lambda,T) Q_i(\lambda)\mathrm{d} \lambda\,,$$

where $A$, and $\Omega$ are the area and solid angle enclosed by the pixel, respectively. $\tau$ and $G$ are the eposure time and conversion gain (in ADC units per electron) of the pixel, respectively. $Q_i(\lambda)$ is the quantum efficiency function for color channel $i$ as a function of wavelength.

TODO: Explain the Bayesian inference model to estimate the temperature and the goodness-of-fit indicator.

TODO: Explain what physical parameters of the camera are needed for the estimation process. List the numbers used for the sony IMX273 sensor used in the example code.

## Installation
The package can be installed using `pip` directly from the repo.

TODO: Give detailed instructions and make the repo `pip` installable.

## Usage
TODO: Explain the API and a minimal example. Point to the example script.