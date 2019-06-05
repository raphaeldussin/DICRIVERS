import pandas as pd
import numpy as np
import xarray as xr
from dicrivers.geo_utils import find_closest_ocean_cell_to_river_mouth
from dicrivers.geo_utils import create_plume


def make_bgc_river_input(river_df, variables,
                         lon_grid, lat_grid, mask_grid,
                         lon_mouth_name='mouth_lon',
                         lat_mouth_name='mouth_lat',
                         nitermax=1000,
                         method='average',
                         prox=200.):
    """ create river bgc concentration file
    Parameters
    ----------
    river_df : pandas.DataFrame
        dataframe of rivers including values for the bgc concentrations
        we want to specify and the rspread (radius of spreading)
    variables : list of string
        bgc variables to work on
    lon_grid : numpy.ndarray
        longitudes of the output grid
    lat_grid : numpy.ndarray
        latitudes of the output grid
    mask_grid : numpy.ndarray
        land/sea mask of the output grid
    lon_mouth_name : string
        name of river mouth longitude in dataframe
    lat_mouth_name : string
        name of river mouth latitude in dataframe
    nitermax : integer
        maximum iterations allowed for convergence
    method : string
        merging method for plumes (available: 'average')
    prox : float
        acceptable maximum distance (in km) between gridcell and true
        river location
    Returns
    -------
    river_conc : xarray.Dataset
        gridded concentrations of bgc variables
    """
    assert(isinstance(river_df, pd.DataFrame))
    assert(isinstance(variables, list))
    assert(isinstance(lon_grid, np.ndarray))
    assert(isinstance(lat_grid, np.ndarray))
    assert(isinstance(mask_grid, np.ndarray))

    # test mak_grid is 2d
    if len(mask_grid.shape) != 2:
        raise IOError('mask must be 2d')
    # test presence of rspread
    if 'rspread' not in river_df.keys():
        raise IOError('river dataframe must include radius of spreading')

    # in order to do what we want, we need to move to xarray framework
    ny, nx = mask_grid.shape
    nriver = len(river_df)
    river_plumes = xr.DataArray(np.zeros((nriver, ny, nx)),
                                coords={'river': np.arange(nriver),
                                        'lat': (['y', 'x'], lat_grid),
                                        'lon': (['y', 'x'], lon_grid)},
                                dims=['river', 'y', 'x'])

    for index, river in river_df.iterrows():
        # find the closest ocean point to river mouth
        lon_river = river[lon_mouth_name]
        lat_river = river[lat_mouth_name]
        jmouth, imouth = find_closest_ocean_cell_to_river_mouth(lon_river,
                                                                lat_river,
                                                                lon_grid,
                                                                lat_grid,
                                                                mask_grid,
                                                                prox=prox)
        # create the plume (mask for this particular river)
        if (jmouth is not None) and (imouth is not None):
            plume = create_plume(imouth, jmouth, lon_grid, lat_grid, mask_grid,
                                 rspread=river['rspread'], nitermax=nitermax)

            river_plumes.loc[{'river': index}] = plume

    # add dataarray to Dataset
    ds_rivers = xr.Dataset({'river_plumes': river_plumes})
    # populate dataset with concentration values for each variable
    for var in variables:
        ds_rivers.update({var: (['river'], river_df[var])})

    # merge the plumes and concentrations
    river_conc = xr.Dataset()
    for var in variables:
        if method == 'average':
            merged_conc = merge_average(ds_rivers, var)
        else:
            raise ValueError('only method available yet is average')
        river_conc.update({var: merged_conc})

    return river_conc


def merge_average(ds, var):
    merged = (ds[var] * ds['river_plumes']).sum(dim='river') / \
              ds['river_plumes'].sum(dim='river')
    # when the is no plumes, this returns a NaN, so we fix a poteriori
    merged = merged.fillna(0)
    return merged
