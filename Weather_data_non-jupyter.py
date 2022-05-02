from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
from netCDF4 import num2date
import numpy as np
import scipy.ndimage as ndimage
from siphon.ncss import NCSS
from pyproj import Proj, transform
import utm
from warnings import filterwarnings
from scipy.interpolate import griddata
from scipy.interpolate import interpn

# Helper function for finding proper time variable
def find_time_var(var, time_basename='time'):
    for coord_name in var.coordinates.split():
        if coord_name.startswith(time_basename):
            return coord_name
    raise ValueError('No time variable found for ' + var.name)

base_url = 'https://www.ncei.noaa.gov/thredds/ncss/model-rap130anl-old/'
dt = datetime(2016, 1, 1, 12)
ncss = NCSS('{}{dt:%Y%m}/{dt:%Y%m%d}/rap_130_{dt:%Y%m%d}'
            '_{dt:%H}00_000.grb2'.format(base_url, dt=dt))

# Create lat/lon box for location you want to get data for
query = ncss.query().time(dt)
query.lonlat_box(north=36.5, south=25.8, east=-93.5, west=-106.65)
query.accept('netcdf')

# Request data for model "surface" data
query.variables('Temperature_height_above_ground',
                'u-component_of_wind_height_above_ground',
                'v-component_of_wind_height_above_ground')
data = ncss.get_data(query)

data.variables.keys()

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
x_2d, y_2d = np.meshgrid(x, y)

points = (x,y)

# Get subset of valid points for which we have data
x_valid_temp = x_2d[~temp_80m.mask].ravel()
x_valid_u = x_2d[~u_wind_80m.mask].ravel()
x_valid_v = x_2d[~v_wind_80m.mask].ravel()

y_valid_temp = y_2d[~temp_80m.mask].ravel()
y_valid_u = y_2d[~u_wind_80m.mask].ravel()
y_valid_v = y_2d[~v_wind_80m.mask].ravel()

point_temp = (x_valid_temp,y_valid_temp)
point_u = (x_valid_u,y_valid_u)
point_v = (x_valid_v,y_valid_v)

temp_80m_valid = temp_80m[~temp_80m.mask].squeeze()
u_wind_valid = u_wind_80m[~u_wind_80m.mask].squeeze()
v_wind_valid = v_wind_80m[~v_wind_80m.mask].squeeze()

# Convert masked weather arrays to numpy arrays by interpolating to fill missing data points
temp_80m_interp = interpn(point_temp,temp_80m_valid,points, method='nearest')
u_80m_interp = interpn(point_u,u_wind_valid,points, method='nearest')
v_80m_interp = griddata(point_v,v_wind_valid,points, method='nearest')