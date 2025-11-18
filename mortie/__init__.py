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
    geo2mort,  # Import the actual geo2mort function
    # New inverse functions
    infer_order_from_morton,
    validate_morton,
    mort2norm,
    norm2uniq,
    uniq2geo,
    mort2geo,
    mort2bbox,
    mort2polygon,
)

__all__ = [
    'tools',
    'geo2mort',
    'mort2geo',
    'mort2bbox',
    'mort2polygon',
    'infer_order_from_morton',
    'validate_morton',
    'mort2norm',
    'norm2uniq',
    'uniq2geo',
    'order2res',
    'res2display',
    'unique2parent',
    'heal_norm',
    'VaexNorm2Mort',
    'fastNorm2Mort',
    'geo2uniq',
    'clip2order',
]

# The Rust extension is imported and used internally by fastNorm2Mort in tools.py
# No need to do anything here - tools.py handles the Rust integration
