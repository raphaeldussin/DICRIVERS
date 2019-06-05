import pandas as pd
import os
import numpy as np
import xarray as xr
# requires pytest-datafiles


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data/',
    )


def test_make_bgc_river_input(datafiles):
    'unit test'
    from dicrivers import make_bgc_river_input
    rivers = pd.read_csv(FIXTURE_DIR + '20major_rivers.csv')

    # -------------------------------------------------------------------------
    # regular global grid 0 - 360
    lon_grid, lat_grid = np.meshgrid(np.arange(0, 360, 1),
                                     np.arange(-90, 90, 1))
    mask_grid = np.ones(lon_grid.shape)

    variables = ['testvar']
    out = make_bgc_river_input(rivers, variables,
                               lon_grid, lat_grid, mask_grid,
                               lon_mouth_name='mouth_lon',
                               lat_mouth_name='mouth_lat',
                               nitermax=1000,
                               method='average')

    assert(isinstance(out, xr.Dataset))
    assert(isinstance(out['testvar'], xr.DataArray))
    assert(out['testvar'].min().values == 0)
    assert(out['testvar'].max().values == 99)

    # -------------------------------------------------------------------------
    # regular regional grid
    lon_grid, lat_grid = np.meshgrid(np.arange(-100, 30, 1),
                                     np.arange(20, 70, 1))
    mask_grid = np.ones(lon_grid.shape)

    variables = ['testvar']
    out = make_bgc_river_input(rivers, variables,
                               lon_grid, lat_grid, mask_grid,
                               lon_mouth_name='mouth_lon',
                               lat_mouth_name='mouth_lat',
                               nitermax=1000,
                               method='average')

    assert(isinstance(out, xr.Dataset))
    assert(isinstance(out['testvar'], xr.DataArray))
    assert(out['testvar'].min().values == 0)
    assert(out['testvar'].max().values == 96)
