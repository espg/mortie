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
    geo2mort,
    # Inverse functions
    infer_order_from_morton,
    validate_morton,
    mort2norm,
    norm2uniq,
    uniq2geo,
    mort2geo,
    mort2bbox,
    mort2polygon,
    generate_morton_children,
    mort2healpix,
    morton_buffer,
)

# Import coverage functions
from .coverage import morton_coverage

# Import prefix trie functions
from .prefix_trie import (
    MortonChild,
    split_children,
    split_children_geo,
    geo_morton_polygon,
    morton_polygon,
    morton_polygon_from_array,
)

__all__ = [
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
    'generate_morton_children',
    'mort2healpix',
    'morton_buffer',
    'morton_coverage',
    'MortonChild',
    'split_children',
    'split_children_geo',
    'geo_morton_polygon',
    'morton_polygon',
    'morton_polygon_from_array',
]

# The Rust extension is imported and used internally by fastNorm2Mort in tools.py
# No need to do anything here - tools.py handles the Rust integration
