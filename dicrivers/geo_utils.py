import numpy as np
import scipy.ndimage as si


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
        acceptable maximum distance (in km) between gridcell and true
        river location
    Returns
    -------
    jmouth, imouth : integer
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

    jmouth, imouth = np.unravel_index(arc.argmin(), lon_grid.shape)
    jmouth = int(jmouth)
    imouth = int(imouth)
    # check that we don't fall on a land cell
    assert(mask_grid[jmouth, imouth] == 1)
    # filter out points that do not belong to lat_grid,
    # i.e. are farther away than a few grid cells
    threshold = proximity_range / rearth
    if arc.min() > threshold:  # proximity threshold exceeded
        jmouth = None
        imouth = None
    return jmouth, imouth


def create_plume(imouth, jmouth, lon_grid, lat_grid, mask_grid,
                 rspread=10, nitermax=1000):
    """ create the plume for the river at imouth, jmouth with selected
    spreading.
    Parameters
    ----------
    imouth : integer
        index of river mouth in x
    jmouth : integer
        index of river mouth in y
    lon_grid : np.array
        longitude of ocean grid
    lat_grid : np.array
        latitude of ocean grid
    mask_grid : np.array
        land/sea mask of ocean grid
    rspread : integer
        number of gridpoints for spreading the plume
    nitermax : integer
        maximum number of iterations for spreading algo
    Returns
    -------
    plume : np.array
        binary mask of same dimensions of mask_grid that is equal to one
        where the plume is, zero elsewhere.
    """

    #  initialize plume
    plume = np.zeros(mask_grid.shape)
    plume[jmouth, imouth] = 1

    #  for computational efficiency, we used subsets
    # to +/- rspread in both directions
    imin = max(0, imouth - rspread)
    jmin = max(0, jmouth - rspread)
    imax = min(imouth+rspread+1, -1)
    jmax = min(jmouth+rspread+1, -1)

    mask_zoom = mask_grid[jmin:jmax, imin:imax]
    plume_zoom = plume[jmin:jmax, imin:imax]

    #  run an iterative loop to spread the plume
    #  init array
    plume_zoom_old = plume_zoom.copy()

    for kk in np.arange(nitermax):
        #  use binary dilatation to spread plume
        plume_zoom_new = si.binary_dilation(plume_zoom_old)
        #  correct spreading with land/sea mask
        plume_zoom_new = plume_zoom_new * mask_zoom
        #  stop if we converged
        if (plume_zoom_new == plume_zoom_old).all():
            break
        # update array
        plume_zoom_old = plume_zoom_new.copy()

        # go back to full size array
        plume[jmin:jmax, imin:imax] = plume_zoom_new

    return plume
