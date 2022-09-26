"""
functions for morton indexing

The unit test should take a coordinate list/array of a few cities (~10 is fine, but make sure that they're distributed in the North and South Hemispheres, and across a bunch of countries), run them through the existing fastNorm2Mort function, and verify that the output is the same as whatever your norm2mort function produces.
"""

from numba import int64, vectorize
import healpy as hp
import numpy as np

@vectorize([int64(int64, int64, int64)])
def fastNorm2Mort(order, normed, parents):
    # General version, for arbitrary order
    if order > 18:
        raise ValueError("Max order is 18 (to output to 64-bit int).")
    mask = np.int64(3*4**(order-1))
    num = 0
    for j, i in enumerate(range(order, 0, -1)):
        nextBit = (normed & mask) >> ((2*i) - 2)
        num += (nextBit+1) * 10**(i-1)
        mask = mask >> 2
    if parents is not None:
        if parents >= 6:
            parents = parents - 11
            parents = parents * 10**(order)
            num = num + parents
            num = -1 * num
            num = num - (6 * 10**(order))
        else:
            parents = (parents + 1) * 10**(order)
            num = num + parents
    return num
