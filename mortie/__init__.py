"""
mortie: a library for generating morton indices
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mortie")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = [
        'tools',
        'geo2mort',
        'mort2geo',
        ]

# Import Rust-accelerated functions
try:
    import rustie
    # Alias the Rust function to the expected Python API
    geo2mort = rustie.fast_norm2mort
    # mort2geo not yet implemented in Rust
    mort2geo = None
except (ImportError, AttributeError):
    # Fallback: Rust extension not available
    geo2mort = None
    mort2geo = None
