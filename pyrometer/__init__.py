"""
pyrometer -- ratio-pyrometry temperature estimation from colour images.

Public API
----------
RatioPyrometer
    Main class: initialise with quantum-efficiency data, call `.analyze()`
    on an image path or array to get a posterior temperature estimate.
PyrometerResult
    Dataclass returned by `.analyze()`.
plot_analysis
    Generate the standard 3 x 3 analysis figure from a PyrometerResult.
"""

from .core import PyrometerResult, RatioPyrometer
from .plotting import plot_analysis

__all__ = ["RatioPyrometer", "PyrometerResult", "plot_analysis"]
