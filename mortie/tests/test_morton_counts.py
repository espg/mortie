import pytest
import numpy as np
import pandas as pd
import healpy as hp

def test_healpy_install():
    nside = 2**6
    b2idx = hp.ang2pix(nside,132.85, 77.32, lonlat=True) 
    assert type(b2idx) == np.int64

