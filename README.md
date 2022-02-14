mortie
======

Mortie is a library for applying morton indexing to healpix grids. Morton
numbering (also called z-ordering) facilitates several geospatial operators
such as buffering and neighborhood look-ups, and can generally be thought of as
a type of geohashing.

This particular implementation focuses on hierarchical healpix maps, and is
mostly inspired from this paper.

Dependencies currently are numpy, numba, and mhealpy (which itself has healpy
as a dependency). Ideally, these will be reduced to just healpy and numpy in
the near future. Although not a dependency, there are several functions that
have been written to interface with the vaex project. 

Initial funding of this work was supported by the ICESat-2 project science
office, at the Laboratory for Cryospheric Sciences (NASA Goddard, Section 615). 
