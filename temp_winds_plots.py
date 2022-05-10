"""
=============================
Model Surface Output
=============================

Plot an surface map with mean sea level pressure (MSLP),
2m Temperature (F), and Wind Barbs (kt).

"""
#%% Imports
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
from netCDF4 import num2date
import numpy as np
import scipy.ndimage as ndimage
from siphon.ncss import NCSS
from powersimdata.input.grid import Grid

# Helper functions

# Helper function for finding proper time variable
def find_time_var(var, time_basename='time'):
    for coord_name in var.coordinates.split():
        if coord_name.startswith(time_basename):
            return coord_name
    raise ValueError('No time variable found for ' + var.name)

#%% Create NCSS object to access the NetcdfSubset
# Data from NCEI GFS 0.5 deg Analysis Archive
# Find bounding box for texas interconnection
grid = Grid(["Texas"])
buses = grid.bus
north_box = buses.lat.max() + 1
south_box = buses.lat.min() - 1
west_box = buses.lon.min() - 1
east_box = buses.lon.max() + 1

base_url = 'https://www.ncei.noaa.gov/thredds/ncss/model-rap130anl-old/'
dt = datetime(2016, 1, 1, 2)
ncss = NCSS('{}{dt:%Y%m}/{dt:%Y%m%d}/rap_130_{dt:%Y%m%d}'
            '_{dt:%H}00_000.grb2'.format(base_url, dt=dt))

# Create lat/lon box for location you want to get data for
query = ncss.query().time(dt)
query.lonlat_box(north=north_box, south=south_box, east=east_box, west=west_box)
query.accept('netcdf')
query.add_lonlat(value=True)

# Request data for model "surface" data
query.variables('Temperature_height_above_ground',
                'u-component_of_wind_height_above_ground',
                'v-component_of_wind_height_above_ground')
data = ncss.get_data(query)

# Pull out variables you want to use
temp = units.K * data.variables['Temperature_height_above_ground'][:].squeeze()
lev_80m_temp = np.where(data.variables['height_above_ground1'][:] == 80)[0][0]
temp_80m = temp[lev_80m_temp]
u_wind = units('m/s') * data.variables['u-component_of_wind_height_above_ground'][:].squeeze()
v_wind = units('m/s') * data.variables['v-component_of_wind_height_above_ground'][:].squeeze()
time_var = data.variables[find_time_var(data.variables['Temperature_height_above_ground'])]
lat = data.variables['lat'][:].squeeze()
lon = data.variables['lon'][:].squeeze()

# Convert number of hours since the reference time into an actual date
time = num2date(time_var[:].squeeze(), time_var.units)

lev_80m_wind = np.where(data.variables['height_above_ground4'][:] == 80)[0][0]
u_wind_80m = u_wind[lev_80m_wind]
v_wind_80m = v_wind[lev_80m_wind]

# Combine 1D latitude and longitudes into a 2D grid of locations
# lon_2d, lat_2d = np.meshgrid(lon, lat)
lon_2d, lat_2d = lon, lat

# Smooth MSLP a little
# Be sure to only put in a 2D lat/lon or Y/X array for smoothing
# smooth_temp = ndimage.gaussian_filter(temp_80m, sigma=3, order=0)
# smooth_u = ndimage.gaussian_filter(u_wind_80m, sigma=3, order=0) 
# smooth_v = ndimage.gaussian_filter(v_wind_80m, sigma=3, order=0) 
smooth_temp = temp_80m
smooth_u = u_wind_80m 
smooth_v = v_wind_80m 

# Set Projection of Data
datacrs = ccrs.PlateCarree()

# Set Projection of Plot
cent_lat = (north_box+south_box)/2.0
cent_lon = (east_box+west_box)/2.0
plotcrs = ccrs.LambertConformal(central_latitude=cent_lat, central_longitude=cent_lon)

# Create new figure
fig = plt.figure(figsize=(11, 8.5))

# Add the map and set the extent
ax = plt.subplot(111, projection=plotcrs)
plt.title('Wind at 80 m')
ax.set_extent([west_box, east_box-0.8, south_box, north_box])

# Plot 80m Temperature Contours
# clevtemp = np.arange(-60, 101, 10)
# cs = ax.contourf(lon_2d, lat_2d, smooth_temp,levels=100, cmap='viridis',transform=datacrs)
# fig.colorbar(cs, ax=ax)

wind_mag = np.sqrt(u_wind_80m.magnitude**2 + v_wind_80m.magnitude**2)

# Plot 80m Wind Barbs

cs = ax.contourf(lon_2d, lat_2d, wind_mag, levels=100, cmap='twilight_shifted',transform=datacrs)
fig.colorbar(cs, ax=ax)

# Add state boundaries to plot
states_provinces = cfeature.NaturalEarthFeature(category='cultural',
                                                name='admin_1_states_provinces_lakes',
                                                scale='50m', facecolor='none')
ax.add_feature(states_provinces, edgecolor='black', linewidth=1)

# Add country borders to plot
country_borders = cfeature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_countries',
                                               scale='50m', facecolor='none')
ax.add_feature(country_borders, edgecolor='black', linewidth=1)

ax.barbs(lon_2d, lat_2d, u_wind_80m.magnitude, v_wind_80m.magnitude,
         length=6, regrid_shape=20, pivot='middle', transform=datacrs)
         # barbcolor='w',

plt.show()

# %%
