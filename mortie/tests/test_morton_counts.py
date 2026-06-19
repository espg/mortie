import pytest
import numpy as np
from mortie._healpix import ang2pix

def test_healpix_via_rust():
    # ang2pix is depth-native: pass the HEALPix order (depth), not nside.
    depth = 6
    b2idx = ang2pix(depth, 132.85, 77.32)
    assert type(b2idx) == np.int64
