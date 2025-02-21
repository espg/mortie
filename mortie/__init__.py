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
        'tools'
        ]

from mortie import *
