import pandas as pd
import os
import numpy as np
# requires pytest-datafiles


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data/',
    )


def test_find_closest_ocean_cell_to_river_mouth(datafiles):
    'unit test'
    from dicrivers.geo_utils import find_closest_ocean_cell_to_river_mouth
    rivers = pd.read_csv(FIXTURE_DIR + '20major_rivers.csv')
    print(rivers)

    # -------------------------------------------------------------------------
    # regular global grid 0 - 360
    lon_grid, lat_grid = np.meshgrid(np.arange(0, 360, 1),
                                     np.arange(-90, 90, 1))
    mask_grid = np.ones(lon_grid.shape)

    for index, river in rivers.iterrows():
        print(river)
        print(river['mouth_lon'], river['mouth_lat'])
        lon_river = river['mouth_lon']
        lat_river = river['mouth_lat']
        jmouth, imouth = find_closest_ocean_cell_to_river_mouth(lon_river,
                                                                lat_river,
                                                                lon_grid,
                                                                lat_grid,
                                                                mask_grid)
        print(jmouth, imouth)
        # grid resolution is one degree, error should be < 1 deg
        lon_river_360 = np.mod(lon_river+360, 360)
        assert(np.abs(lon_grid[jmouth, imouth] - lon_river_360) < 1)
        assert(np.abs(lat_grid[jmouth, imouth] - lat_river) < 1)

    # -------------------------------------------------------------------------
    # regular global grid 0 - 180
    lon_grid, lat_grid = np.meshgrid(np.arange(-180, 180, 1),
                                     np.arange(-90, 90, 1))
    mask_grid = np.ones(lon_grid.shape)

    for index, river in rivers.iterrows():
        print(river)
        print(river['mouth_lon'], river['mouth_lat'])
        lon_river = river['mouth_lon']
        lat_river = river['mouth_lat']
        jmouth, imouth = find_closest_ocean_cell_to_river_mouth(lon_river,
                                                                lat_river,
                                                                lon_grid,
                                                                lat_grid,
                                                                mask_grid)
        print(jmouth, imouth)
        # grid resolution is one degree, error should be < 1 deg
        assert(np.abs(lon_grid[jmouth, imouth] - lon_river) < 1)
        assert(np.abs(lat_grid[jmouth, imouth] - lat_river) < 1)

    # -------------------------------------------------------------------------
    # regional grid
    lon_grid, lat_grid = np.meshgrid(np.arange(-120, 30, 0.25),
                                     np.arange(10, 70, 0.25))
    mask_grid = np.ones(lon_grid.shape)

    # check Mississippi is there
    Mississippi = rivers.loc[rivers['basinname'] == 'Mississippi']
    lon_miss = Mississippi['mouth_lon'].values
    lat_miss = Mississippi['mouth_lat'].values

    jmouth, imouth = find_closest_ocean_cell_to_river_mouth(lon_miss,
                                                            lat_miss,
                                                            lon_grid,
                                                            lat_grid,
                                                            mask_grid)
    assert(jmouth is not None)
    assert(imouth is not None)

    # but amazon is not
    Amazon = rivers.loc[rivers['basinname'] == 'Amazon']
    lon_ama = Amazon['mouth_lon'].values
    lat_ama = Amazon['mouth_lat'].values

    jmouth, imouth = find_closest_ocean_cell_to_river_mouth(lon_ama,
                                                            lat_ama,
                                                            lon_grid,
                                                            lat_grid,
                                                            mask_grid)

    assert(jmouth is None)
    assert(imouth is None)

    # mask still needs testing
    return None


def test_create_plume(datafiles):
    '''unit test'''
    from dicrivers.geo_utils import find_closest_ocean_cell_to_river_mouth
    from dicrivers.geo_utils import create_plume

    rivers = pd.read_csv(FIXTURE_DIR + '20major_rivers.csv')
    # -------------------------------------------------------------------------
    # regular global grid 0 - 180
    lon_grid, lat_grid = np.meshgrid(np.arange(-180, 180, 1),
                                     np.arange(-90, 90, 1))
    mask_grid = np.ones(lon_grid.shape)

    # check Mississippi is there
    Mississippi = rivers.loc[rivers['basinname'] == 'Mississippi']
    lon_miss = Mississippi['mouth_lon'].values
    lat_miss = Mississippi['mouth_lat'].values

    jmouth, imouth = find_closest_ocean_cell_to_river_mouth(lon_miss,
                                                            lat_miss,
                                                            lon_grid,
                                                            lat_grid,
                                                            mask_grid)

    rspread = 10
    nitermax = 1000
    plume = create_plume(imouth, jmouth, lon_grid, lat_grid, mask_grid,
                         rspread=rspread, nitermax=nitermax)

    assert(isinstance(plume, np.ndarray))
    #  test for min/max values in the no-mask case
    assert(plume.sum() > 1)
    assert(plume.sum() < (2*rspread+1)**2 + 1)
    return None
