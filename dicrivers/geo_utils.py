import numpy as np


def find_closest_ocean_cell_to_river_mouth(lon_river, lat_river,
                                           lon_grid, lat_grid, mask_grid,
                                           proximity_range=200.):
    '''find grid's closest ocean grid point to true geographical location
    of river mouth by computing distance to the true location and finding
    minimum of array. Land points are set to very large value to exclude them.
    A proximity_range criteria allows to filter out rivers not in regional
    domain.

    Parameters
    ----------
    lon_river : float
        longitude of the river mouth in real life
    lat_river : float
        latitude of the river mouth in real life
    lon_grid : np.array
        longitude of ocean grid
    lat_grid : np.array
        latitude of ocean grid
    mask_grid : np.array
        land/sea mask of ocean grid
    proximity_range : float
        acceptable maximum distance (in km) between cell and true
        river location
    Returns
    -------
    jcell, icell : integer
        coordinates of river mouth in grid
    '''
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0
    rearth = 6400
    # phi = 90 - latitude
    phi1 = (90.0 - lat_river)*degrees_to_radians
    phi2 = (90.0 - lat_grid)*degrees_to_radians
    # theta = longitude
    theta1 = lon_river*degrees_to_radians
    theta2 = lon_grid*degrees_to_radians
    # Compute spherical distance from spherical coordinates.
    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) +
           np.cos(phi1)*np.cos(phi2))
    arc = np.arccos(cos)

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    arc[np.where(mask_grid == 0)] = 1.e36

    jcell, icell = np.unravel_index(arc.argmin(), lon_grid.shape)
    jcell = int(jcell)
    icell = int(icell)
    # check that we don't fall on a land cell
    assert(mask_grid[jcell, icell] == 1)
    # filter out points that do not belong to lat_grid,
    # i.e. are farther away than a few grid cells
    threshold = proximity_range / rearth
    if arc.min() > threshold:  # proximity threshold exceeded
        jcell = None
        icell = None
    return jcell, icell
