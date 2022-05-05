from datetime import datetime
from cvxpy import length
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
import time

import requests
from bs4 import BeautifulSoup

def avail_hours(Y, M, D):
    dt = datetime(Y, M, D)
    catalog = 'https://www.ncei.noaa.gov/thredds/catalog/model-rap130anl-old/'
    url = '{}{dt:%Y%m}/{dt:%Y%m%d}/catalog.html'.format(catalog, dt=dt)

    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    hrefs = pd.Series([link.get('href') for link in soup.find_all('a')])
    href_df = pd.DataFrame({k: hrefs[hrefs.str.contains('.grb')].str.split('_').str[k].str.split('.').str[0] for k in [-2, -1]})
    hrs = href_df[href_df[-1]=='001'][-2].drop_duplicates().sort_values()
    return list(hrs.str[:2].astype(int))

# Helper function for finding proper time variable
def find_time_var(var, time_basename='time'):
    for coord_name in var.coordinates.split():
        if coord_name.startswith(time_basename):
            return coord_name
    raise ValueError('No time variable found for ' + var.name)

base_url = 'https://www.ncei.noaa.gov/thredds/ncss/model-rap130anl-old/'

# Find bounding box for texas interconnection
grid = Grid(["Texas"])
buses = grid.bus
north_box = buses.lat.max() + 1
south_box = buses.lat.min() - 1
west_box = buses.lon.min() - 1
east_box = buses.lon.max() + 1

months = ['Jul','Aug','Sep','Oct','Nov','Dec']
dates_from = ['2016-7-1','2016-8-1','2016-9-1','2016-10-1','2016-11-1','2016-12-1']
dates_to = ['2016-7-31','2016-8-31','2016-9-30','2016-10-31','2016-11-30','2017-1-1']

for i in range(len(months)):
    # Empty dictionaries to store data at each timestep
    u_all = {}
    v_all = {}
    temp_all = {}
    x_orig_all = {}
    y_orig_all = {}
    times = {}

    # Hour counter
    h = 0

    for date in pd.date_range(start=dates_from[i], end=dates_to[i], freq='D'):
        print(date, end=' ')
        hrs = avail_hours(date.year, date.month, date.day)
        print(len(hrs))
        for hr in hrs:
            dt = pd.to_datetime('{}-{}-{} {}:00'.format(date.year, date.month, date.day, hr))
        
            ncss = NCSS('{}{dt:%Y%m}/{dt:%Y%m%d}/rap_130_{dt:%Y%m%d}'
                        '_{dt:%H}00_000.grb2'.format(base_url, dt=dt))

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
            x_orig = data.variables['x'][:].squeeze()
            y_orig = data.variables['y'][:].squeeze()
            time_var = data.variables[find_time_var(data.variables['Temperature_height_above_ground'])]

            # Convert number of hours since the reference time into an actual date
            time_val = num2date(time_var[:].squeeze(), time_var.units)

            lev_80m_wind = np.where(data.variables['height_above_ground4'][:] == 80)[0][0]
            u_wind_80m = u_wind[lev_80m_wind]
            v_wind_80m = v_wind[lev_80m_wind]

            # # Convert masked arrays to regular numpy arrays
            # x = x_orig.compressed()
            # y = x_orig.compressed()

            # # Combine 1D x and y coordinates into a 2D grid of locations
            # x_2d, y_2d = np.meshgrid(x, y,indexing='ij')

            # # Get subset of valid points for which we have data
            # x_valid_temp = x_2d[~temp_80m.mask].ravel()
            # x_valid_u = x_2d[~u_wind_80m.mask].ravel()
            # x_valid_v = x_2d[~v_wind_80m.mask].ravel()

            # y_valid_temp = y_2d[~temp_80m.mask].ravel()
            # y_valid_u = y_2d[~u_wind_80m.mask].ravel()
            # y_valid_v = y_2d[~v_wind_80m.mask].ravel()

            # temp_80m_valid = temp_80m[~temp_80m.mask].ravel()
            # u_wind_valid = u_wind_80m[~u_wind_80m.mask].ravel()
            # v_wind_valid = v_wind_80m[~v_wind_80m.mask].ravel()

            # points_temp = np.transpose(np.vstack((x_valid_temp,y_valid_temp)))
            # points_u = np.transpose(np.vstack((x_valid_u,y_valid_u)))
            # points_v = np.transpose(np.vstack((x_valid_v,y_valid_v)))

            # # Convert masked weather arrays to numpy arrays by interpolating to fill missing data points
            # # interpolation methods - nearest, linear, cubic
            # temp_80m_interp = griddata(points_temp, temp_80m_valid, (x_2d,y_2d), method='linear')
            # u_80m_interp = griddata(points_u, u_wind_valid, (x_2d,y_2d), method='linear')
            # v_80m_interp = griddata(points_v, v_wind_valid, (x_2d,y_2d), method='linear')

            # # Convert x and y from lambert conformal conic projection to lat-lon
            # x_2d_array = x_2d.ravel()
            # y_2d_array = y_2d.ravel()

            # transformer = Transformer.from_crs('epsg:2154', 'epsg:4326')
            # lat,lon = transformer.transform(x_2d_array,y_2d_array)

            # # Convert lat-lon to utm projection
            # result = utm.from_latlon(lat,lon)
            # x, y = result[0], result[1]

            x_orig_all[h] = x_orig
            y_orig_all[h] = y_orig
            temp_all[h] = temp_80m
            u_all[h] = u_wind_80m
            v_all[h] = v_wind_80m
            times[h] = dt

            h += 1
            time.sleep(5)

    str1 = "C:\\Users\\vinee\\OneDrive - Massachusetts Institute of Technology\\MIT\\Semesters\\Spring 2022\\15.S08\\Project\\weather_data_"
    str2 = "2016.pkl"
    file_name = str1 + months[i] + str2

    with open(file_name, 'wb') as file:
        pickle.dump(x_orig_all, file)
        pickle.dump(y_orig_all, file)
        pickle.dump(temp_all, file)
        pickle.dump(u_all, file)
        pickle.dump(v_all, file)