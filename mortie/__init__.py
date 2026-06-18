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
    morton_buffer_meters,
)

# Import coverage functions
from .coverage import (
    morton_coverage,
    morton_coverage_moc,
    compress_moc,
    moc_to_order,
)
from .linestring import linestring_coverage

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
    'morton_buffer_meters',
    'morton_coverage',
    'morton_coverage_moc',
    'compress_moc',
    'moc_to_order',
    'linestring_coverage',
    'MortonChild',
    'split_children',
    'split_children_geo',
    'geo_morton_polygon',
    'morton_polygon',
    'morton_polygon_from_array',
]

# morton_index datatype (phase 5) + Arrow interop (phase 4) for issue #35. The
# pandas ExtensionArray and the pyarrow ExtensionType are optional extras:
# importing mortie must succeed with only numpy installed, so the names are
# exposed lazily and built only when pandas / pyarrow are present (touching them
# without the extra raises a clear ImportError). See mortie/morton_index.py and
# mortie/arrow.py.
from . import (
    arrow,  # noqa: F401
    morton_index,  # noqa: F401
)

_ARROW_NAMES = (
    "MortonIndexType",
    "MortonIndexExtArray",
    "morton_index_type",
    "from_morton_index",
    "to_morton_index",
)


def __getattr__(name):
    if name in ("MortonIndexDtype", "MortonIndexArray"):
        return getattr(morton_index, name)
    if name in _ARROW_NAMES:
        return getattr(arrow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ += ['MortonIndexDtype', 'MortonIndexArray', 'morton_index']
__all__ += list(_ARROW_NAMES) + ['arrow']

# The Rust extension is imported and used internally by fastNorm2Mort in tools.py
# No need to do anything here - tools.py handles the Rust integration
