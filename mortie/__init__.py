"""mortie: a library for generating morton indices."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mortie")
except PackageNotFoundError:
    # package is not installed
    pass

# MOC algebra -- the kernel layer (split out of coverage by domain, issue #156;
# the module is mortie/_moc.py since issue #196 freed the `moc` name for the
# Moc constructor, but the names stay flat on the package either way).
from ._moc import (
    common_ancestor,
    compress_moc,
    moc_and,
    moc_intersects,
    moc_min,
    moc_minus,
    moc_not,
    moc_or,
    moc_to_order,
    moc_xor,
    split_base_cells,
)

# toc word -- temporal order coverage (issue #175; the module is
# mortie/_toc.py since issue #198 freed the `toc` name for the Toc
# constructor, but the names stay flat on the package either way -- the four
# grid/epoch constants included, now that the submodule spelling is gone).
from ._toc import (
    GPS_EPOCH_NS,
    Q_END_NS,
    Q_START_NS,
    TOC_MAX_NS,
    from_datetime64,
    from_gps_ns,
    span2toc,
    time2toc,
    to_datetime64,
    to_gps_ns,
    toc2time,
    toc_and,
    toc_contains,
    toc_is_range,
    toc_merge,
    toc_normalize,
    toc_overlaps,
    toc_reduce,
)

# The batch-kernel module (issue #170).  Since issue #187 the plural batch
# names retired -- each kernel is private, reached through its polymorphic
# entry point -- and the one surviving public name here is the batch-native
# coverer (its ragged signature has no scalar shape to collapse into).
from .batch import (
    polygons_to_morton_mocs,
)

# Cell-set dilation (split out of tools by domain, issue #159)
from .buffer import (
    morton_buffer,
    morton_buffer_meters,
)

# Address-space conversions (split out of tools by domain, issue #159)
from .convert import (
    authalic_to_geodetic,
    geo2mort,
    geo2uniq,
    geodetic_to_authalic,
    mort2bbox,
    mort2geo,
    mort2healpix,
    mort2norm,
    mort2polygon,
    norm2mort,
    norm2uniq,
    uniq2geo,
    unique2parent,
)

# Import coverage functions.  The scalar morton_coverage_moc retired with the
# plural batch names (issue #187): polygons_to_morton_mocs is the MOC
# coverer's public entry point, and the one-ring-set form is reached through
# from_geometry / from_wkb / from_wkt with moc=True (or Moc).
from .coverage import (
    RingValidity,
    morton_coverage,
    ring_is_simple,
    ring_validity,
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

# The geometry-first object layer over the kernel above (issue #196).  `moc` is
# a callable namespace rather than a submodule: `moc(geojson)` builds a `Moc`,
# and `moc.moc_and`-style attribute access is the deprecation shim for the
# `mortie/moc.py` -> `mortie/_moc.py` rename.
from .moc_object import (
    Moc,
    moc,
)

# Order query/change/validate and the resolution ladder (split out of tools by
# domain, issue #159)
from .orders import (
    clip2order,
    generate_morton_children,
    # Inverse functions
    infer_order_from_morton,
    is_point,
    order2res,
    orders_of,
    orders_of_uniq,
    res2display,
    validate_morton,
)

# Import prefix trie functions
from .prefix_trie import (
    MortonChild,
    geo_morton_polygon,
    morton_polygon,
    morton_polygon_from_array,
    split_children,
    split_children_geo,
)

# Rank-space (x, y) deinterleave for 2-D block views (issue #149)
from .rank_xy import (
    rank_to_xy,
    xy_to_rank,
)

# The temporal object layer over the toc kernel above (issue #198), sibling of
# `Moc`: `Toc` wraps the canonical normalized word set, and `toc` is a
# callable namespace rather than a submodule -- `toc("2020-01-01", ...)`
# builds a `Toc`, and `toc.toc_merge`-style attribute access is the
# deprecation shim for the `mortie/toc.py` -> `mortie/_toc.py` rename.
from .toc_object import (
    Toc,
    toc,
)

__all__ = [
    'geo2mort',
    'geodetic_to_authalic',
    'authalic_to_geodetic',
    'mort2geo',
    'mort2bbox',
    'mort2polygon',
    'infer_order_from_morton',
    'orders_of',
    'orders_of_uniq',
    'is_point',
    'validate_morton',
    'mort2norm',
    'norm2uniq',
    'uniq2geo',
    'order2res',
    'res2display',
    'unique2parent',
    'norm2mort',
    'geo2uniq',
    'clip2order',
    'generate_morton_children',
    'mort2healpix',
    'morton_buffer',
    'morton_buffer_meters',
    'morton_coverage',
    'polygons_to_morton_mocs',
    'RingValidity',
    'ring_is_simple',
    'ring_validity',
    'compress_moc',
    'moc_to_order',
    'moc_or',
    'moc_and',
    'moc_intersects',
    'moc_minus',
    'moc_xor',
    'moc_not',
    'common_ancestor',
    'moc_min',
    'split_base_cells',
    'Moc',
    'moc',
    'Toc',
    'toc',
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
    'rank_to_xy',
    'xy_to_rank',
    'time2toc',
    'span2toc',
    'toc2time',
    'toc_merge',
    'toc_normalize',
    'toc_and',
    'toc_reduce',
    'toc_is_range',
    'toc_overlaps',
    'toc_contains',
    'from_datetime64',
    'to_datetime64',
    'from_gps_ns',
    'to_gps_ns',
    'Q_START_NS',
    'Q_END_NS',
    'TOC_MAX_NS',
    'GPS_EPOCH_NS',
]

# morton_index datatype (phase 5) + Arrow interop (phase 4) for issue #35. The
# pandas ExtensionArray and the pyarrow ExtensionType are optional extras:
# importing mortie must succeed with only numpy installed, so the names are
# exposed lazily and resolved only when pandas / pyarrow are present (touching
# them without the extra raises a clear ImportError). The ExtensionArray classes
# themselves live in mortie/pandas.py, which mortie.morton_index imports on
# demand (issue #135). See mortie/morton_index.py and mortie/arrow.py.
from . import (
    arrow,  # noqa: F401
    morton_index,  # noqa: F401
)

# The decimal parse surface (issue #114). Unlike the ExtensionArray/Arrow names
# below, these need only numpy and the Rust extension, so they are bound
# eagerly rather than through __getattr__ -- and they stay callable (and
# pandas-free) in a numpy-only install, where the lazy names would raise.
from .morton_index import (  # noqa: F401
    MortonWord,
    decimal_to_word,
)

_ARROW_NAMES = (
    "MortonIndexType",
    "MortonIndexExtArray",
    "morton_index_type",
    "from_morton_index",
    "to_morton_index",
)


def __getattr__(name):
    if name == "pandas":
        # mortie.morton_index imports this submodule eagerly when pandas is
        # installed (to register the dtype string), which binds the attribute
        # directly -- so this branch is reached only on a numpy-only install,
        # where `mortie.pandas` must raise the curated ImportError rather than a
        # bare AttributeError. `import_module`, not `from . import pandas`: the
        # latter does a `hasattr` on this package first, re-entering __getattr__.
        return import_module(f"{__name__}.pandas")
    if name in ("MortonIndexDtype", "MortonIndexArray"):
        return getattr(morton_index, name)
    if name in _ARROW_NAMES:
        return getattr(arrow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ += ['MortonIndexDtype', 'MortonIndexArray', 'morton_index']
__all__ += ['MortonWord', 'decimal_to_word']
__all__ += list(_ARROW_NAMES) + ['arrow']
# 'pandas' is deliberately NOT in __all__, unlike the 'morton_index' / 'arrow'
# submodules: `from mortie import *` would then bind the name `pandas` to
# mortie's submodule in the caller's namespace, shadowing the real pandas there.
# `import mortie.pandas` / `mortie.pandas` reach it explicitly (issue #135).

# The Rust extension is imported and used internally by the convert.py encoders
# No need to do anything here - convert.py handles the Rust integration
