"""
mortie: a library for generating morton indices
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mortie")
except PackageNotFoundError:
    # package is not installed
    pass

# Import all Python functions from tools module
from .tools import (
    order2res,
    res2display,
    unique2parent,
    heal_norm,
    VaexNorm2Mort,
    fastNorm2Mort,
    geo2uniq,
    clip2order,
)

__all__ = [
    'tools',
    'geo2mort',
    'mort2geo',
    'order2res',
    'res2display',
    'unique2parent',
    'heal_norm',
    'VaexNorm2Mort',
    'fastNorm2Mort',
    'geo2uniq',
    'clip2order',
]

# Import Rust-accelerated functions
try:
    from . import _rustie
    # Alias the Rust function to the expected Python API
    geo2mort = _rustie.fast_norm2mort
    # mort2geo not yet implemented in Rust
    mort2geo = None
except (ImportError, AttributeError):
    # Fallback: Rust extension not available
    geo2mort = None
    mort2geo = None
