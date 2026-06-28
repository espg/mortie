"""
mortie: a library for generating morton indices
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mortie")
except PackageNotFoundError:
    # package is not installed
    pass

# Import all Python functions from tools module
# Import coverage functions
from .coverage import (
    common_ancestor,
    compress_moc,
    moc_and,
    moc_min,
    moc_minus,
    moc_not,
    moc_or,
    moc_to_order,
    moc_xor,
    morton_coverage,
    morton_coverage_moc,
    split_base_cells,
)
from .geometry import (
    from_geometry,
    from_wkb,
    from_wkt,
    to_geometry,
    to_wkb,
    to_wkt,
)
from .linestring import linestring_coverage

# Import prefix trie functions
from .prefix_trie import (
    MortonChild,
    geo_morton_polygon,
    morton_polygon,
    morton_polygon_from_array,
    split_children,
    split_children_geo,
)
from .tools import (
    clip2order,
    generate_morton_children,
    geo2mort,
    geo2uniq,
    heal_norm,
    # Inverse functions
    infer_order_from_morton,
    mort2bbox,
    mort2geo,
    mort2healpix,
    mort2norm,
    mort2polygon,
    morton_buffer,
    morton_buffer_meters,
    norm2mort,
    norm2uniq,
    order2res,
    res2display,
    uniq2geo,
    unique2parent,
    validate_morton,
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
    'norm2mort',
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
    'moc_or',
    'moc_and',
    'moc_minus',
    'moc_xor',
    'moc_not',
    'common_ancestor',
    'moc_min',
    'split_base_cells',
    'linestring_coverage',
    'from_wkb',
    'from_wkt',
    'from_geometry',
    'to_wkb',
    'to_wkt',
    'to_geometry',
    'geometry',
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

# The Rust extension is imported and used internally by the tools.py encoders
# No need to do anything here - tools.py handles the Rust integration
