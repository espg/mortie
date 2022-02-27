mortie
======

Mortie is a library for applying morton indexing to healpix grids. Morton
numbering (also called z-ordering) facilitates several geospatial operators
such as buffering and neighborhood look-ups, and can generally be thought of as
a type of geohashing.

This particular implementation focuses on hierarchical healpix maps, and is
mostly inspired from [this paper](#1).

TODO:

- [ ] add paper reference
- [ ] add funding information
- [ ] add tests
- [x] remove / prune dead code
- [ ] add example(s)
- [ ] fix north / south bug
- [ ] remove numba dependency
- [ ] update documentation
- [x] publish to pypi

Dependencies currently are numpy, numba, and healpy. Ideally, this will be
reduced to just healpy and numpy in the near future. Although not a dependency,
there are several functions that have been written to interface with the vaex
project. The environment.yaml file contains a full plotting environment needed
to run the examples; requirements.txt are the requirements for only the
library.

Initial funding of this work was supported by the ICESat-2 project science
office, at the Laboratory for Cryospheric Sciences (NASA Goddard, Section 615). 

## References
<a id="1">[1]</a> 
Youngren, Robert W., and Mikel D. Petty. 
"A multi-resolution HEALPix data structure for spherically mapped point data." 
Heliyon 3.6 (2017): e00332. [doi: 10.1016/j.heliyon.2017.e00332](https://doi.org/10.1016/j.heliyon.2017.e00332)
