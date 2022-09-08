"""
functions for morton indexing
"""

from numba import int64, vectorize
import healpy as hp
import numpy as np


def order2res(order):
    res = 111 * 58.6323 * .5**order
    return res


def res2display():
    '''prints resolution levels'''
    for res in range(20):
        print(str(order2res(res)) + ' km at tessellation order ' + str(res))


def unique2parent(unique):
    '''
    Assumes input is UNIQ
    Currently only works on single resolution
    Returns parent base cell
    '''
    orders = np.log2(np.array(unique)/4.0)//2.0
    # this is such an ugly hack-- does little, will blow up with multi res
    orders_ = np.unique(orders)
    if len(orders_) == 1:
        order = int(orders_[0])
    else:
        raise NotImplementedError("Cannot parse mixed resolution unique cells")
    unique = unique // 4**(order-1)
    parent = (unique - 16) // 4
    return parent


def heal_norm(base, order, addr_nest):
    N_pix = hp.order2nside(order)**2
    addr_norm = addr_nest - (base * N_pix)
    return addr_norm


@vectorize([int64(int64, int64)])
def VaexNorm2Mort(normed, parents):
    # Need to use vaex apply;
    # since we can't pass in 'order', we hard code it
    # ...useful for free multithreading...
    order = 18
    mask = np.int64(3*4**(order-1))
    num = 0
    for j, i in enumerate(range(order, 0, -1)):
        nextBit = (normed & mask) >> ((2*i) - 2)
        num += (nextBit+1) * 10**(i-1)
        mask = mask >> 2
    parents = parents - 6
    parents = parents * 10**(order)
    num = num + parents
    return num


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


def geo2uniq(lats, lons, order=18):
    """Calculates UNIQ coding for lat/lon

    Defaults to max morton resolution of order 18"""

    nside = 2**order

    nest = hp.ang2pix(nside, lons, lats, lonlat=True, nest=True)
    uniq = 4 * (nside**2) + nest

    return uniq


def geo2mort(lats, lons, order=18):
    """Calculates morton indices from geographic coordinates

    lats: array-like
    lons: array-like
    order: int"""


    uniq = geo2uniq(lats, lons, order)
    parents = unique2parent(uniq)
    normed = heal_norm(parents, order, uniq)
    morton = fastNorm2Mort(order, normed.ravel(), parents.ravel())

    return morton


def clip2order(clip_order, midx=None, print_factor=False):
    """Convenience function to clip max res morton indices to lower res

    clip_order: int ; resolution to degrade to
    midx: array(ints) or None ; morton indices at order 18

    See `res2display` for approximate resolutions

    Setting print_factor to True will return scaling factor;
    default setting of false will execute the clip on the array"""

    factor = 18 - clip_order

    if print_factor:
        return 10**factor
    else:
        negidx = midx < 0
        clipped = np.abs(midx) // 10**factor
        clipped[negidx] *= -1
        return clipped
