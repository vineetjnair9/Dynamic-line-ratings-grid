from metpy.units import units
from netCDF4 import num2date
import numpy as np
from siphon.ncss import NCSS
import utm
from warnings import filterwarnings
from scipy.interpolate import griddata
from pyproj import Transformer
import pandas as pd
from powersimdata.input.grid import Grid
import pickle
from urllib.error import HTTPError

# Helper function for finding proper time variable
def find_time_var(var, time_basename='time'):
    for coord_name in var.coordinates.split():
        if coord_name.startswith(time_basename):
            return coord_name
    raise ValueError('No time variable found for ' + var.name)

# times = pd.date_range(start='1/1/2016', end='1/1/2017',freq='H')
times = pd.date_range(start='1/1/2016', end='1/2/2016',freq='H')
num_periods = times.size

# Empty dictionaries to store data at each timestep
x_all = {}
y_all = {}
u_all = {}
v_all = {}
temp_all = {}

base_url = 'https://www.ncei.noaa.gov/thredds/ncss/model-rap130anl-old/'

# Find bounding box for texas interconnection
grid = Grid(["Texas"])
buses = grid.bus
north_box = buses.lat.max() + 1
south_box = buses.lat.min() - 1
west_box = buses.lon.min() - 1
east_box = buses.lon.max() + 1

# Loop through all times
for i in range(num_periods):

    # dt = datetime(2016, 1, 1, 12) # Y, M, D, H
    dt = times[i]

    try: 
        ncss = NCSS('{}{dt:%Y%m}/{dt:%Y%m%d}/rap_130_{dt:%Y%m%d}'
                '_{dt:%H}00_000.grb2'.format(base_url, dt=dt))
    except HTTPError: # Just skip this hour & go to the next if not available
        continue

    # Create lat/lon box for location you want to get data for
    query = ncss.query().time(dt)
    query.lonlat_box(north=north_box, south=south_box, east=east_box, west=west_box)
    query.accept('netcdf')

    # Request data for model "surface" data
    query.variables('Temperature_height_above_ground',
                    'u-component_of_wind_height_above_ground',
                    'v-component_of_wind_height_above_ground')
    data = ncss.get_data(query)

    filterwarnings("ignore", category=DeprecationWarning) 

    # Pull out variables you want to use
    temp = units.K * data.variables['Temperature_height_above_ground'][:].squeeze()
    lev_80m_temp = np.where(data.variables['height_above_ground1'][:] == 80)[0][0]
    temp_80m = temp[lev_80m_temp]
    u_wind = units('m/s') * data.variables['u-component_of_wind_height_above_ground'][:].squeeze()
    v_wind = units('m/s') * data.variables['v-component_of_wind_height_above_ground'][:].squeeze()
    x = data.variables['x'][:].squeeze()
    y = data.variables['y'][:].squeeze()
    time_var = data.variables[find_time_var(data.variables['Temperature_height_above_ground'])]

    # Convert number of hours since the reference time into an actual date
    time = num2date(time_var[:].squeeze(), time_var.units)

    lev_80m_wind = np.where(data.variables['height_above_ground4'][:] == 80)[0][0]
    u_wind_80m = u_wind[lev_80m_wind]
    v_wind_80m = v_wind[lev_80m_wind]

    # Convert masked arrays to regular numpy arrays
    x = x.compressed()
    y = y.compressed()

    # Combine 1D x and y coordinates into a 2D grid of locations
    x_2d, y_2d = np.meshgrid(x, y,indexing='ij')

    # Get subset of valid points for which we have data
    x_valid_temp = x_2d[~temp_80m.mask].ravel()
    x_valid_u = x_2d[~u_wind_80m.mask].ravel()
    x_valid_v = x_2d[~v_wind_80m.mask].ravel()

    y_valid_temp = y_2d[~temp_80m.mask].ravel()
    y_valid_u = y_2d[~u_wind_80m.mask].ravel()
    y_valid_v = y_2d[~v_wind_80m.mask].ravel()

    temp_80m_valid = temp_80m[~temp_80m.mask].ravel()
    u_wind_valid = u_wind_80m[~u_wind_80m.mask].ravel()
    v_wind_valid = v_wind_80m[~v_wind_80m.mask].ravel()

    points_temp = np.transpose(np.vstack((x_valid_temp,y_valid_temp)))
    points_u = np.transpose(np.vstack((x_valid_u,y_valid_u)))
    points_v = np.transpose(np.vstack((x_valid_v,y_valid_v)))

    # Convert masked weather arrays to numpy arrays by interpolating to fill missing data points
    # interpolation methods - nearest, linear, cubic
    temp_80m_interp = griddata(points_temp, temp_80m_valid, (x_2d,y_2d), method='linear')
    u_80m_interp = griddata(points_u, u_wind_valid, (x_2d,y_2d), method='linear')
    v_80m_interp = griddata(points_v, v_wind_valid, (x_2d,y_2d), method='linear')

    # Convert x and y from lambert conformal conic projection to lat-lon
    x_2d_array = x_2d.ravel()
    y_2d_array = y_2d.ravel()

    transformer = Transformer.from_crs('epsg:2154', 'epsg:4326')
    lat,lon = transformer.transform(x_2d_array,y_2d_array)

    # Convert lat-lon to utm projection
    result = utm.from_latlon(lat,lon)
    x, y = result[0], result[1]

    x_all[i] = x
    y_all[i] = y
    temp_all[i] = temp_80m_interp
    u_all[i] = u_80m_interp
    v_all[i] = v_80m_interp

with open(r"C:\Users\vinee\Documents\DLR-15s08-project\weather_data_2016.pkl", 'wb') as file:
    pickle.dump(times, file)
    pickle.dump(x_all, file)
    pickle.dump(y_all, file)
    pickle.dump(temp_all, file)
    pickle.dump(u_all, file)
    pickle.dump(v_all, file)