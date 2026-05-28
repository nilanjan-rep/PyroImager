"""
pyrometer – ratio-pyrometry temperature estimation from colour images.

Public API
----------
RatioPyrometer
    Main class: initialise with quantum-efficiency data, call `.analyze()`
    on an image path or array to get per-pixel temperature estimates.
PyrometerResult
    Dataclass returned by `.analyze()`.
"""

from .core import PyrometerResult, RatioPyrometer

__all__ = ["RatioPyrometer", "PyrometerResult"]
