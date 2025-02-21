import pytest
import numpy as np
import pandas as pd
import healpy as hp

basins = pd.read_csv('./Ant_Grounded_DrainageSystem_Polygons.txt',
                     names=['Lat','Lon','basin'], delim_whitespace=True)
b4 = basins[basins.basin == 4]

def test_healpy_install():
    nside = 2**6
    b2idx = hp.ang2pix(nside,b4.Lon.values, b4.Lat.values, lonlat=True) 
    subset = np.unique(b2idx).ravel()
    assert len(subset) == 39

