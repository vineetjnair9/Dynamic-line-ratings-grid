#%% IMPORTS
from powersimdata.input.grid import Grid
from prereise.gather.winddata.rap import rap, helpers # impute

import pandas as pd
from matplotlib import pyplot
import numpy as np
import utm
import math
import pickle
from pyproj import Transformer
import utm

from scipy.spatial.distance import cdist

grid = Grid(["Texas"])

buses = grid.bus

#%%  X and Y coordinates of buses
# Convert from WGS-84 (lat, lon) --> UTM projection
result = utm.from_latlon(buses.lat.to_numpy(),buses.lon.to_numpy())
buses['X'], buses['Y'] = result[0], result[1]

branches = grid.branch

# Assign start and end X and Y coordinates
from_result = utm.from_latlon(branches.from_lat.to_numpy(),branches.from_lon.to_numpy())
branches['fromX'], branches['fromY'] = from_result[0], from_result[1]

to_result = utm.from_latlon(branches.to_lat.to_numpy(),branches.to_lon.to_numpy())
branches['toX'], branches['toY'] = to_result[0], to_result[1]

# Merge lists of x and y coordinates into list of coordinate pair tuples
def merge(list1, list2):
    return list(map(lambda x, y: (x, y), list1, list2))


def sample_points(x1, y1, x2, y2, d):
    # Start: (x1,y1)
    # End: (x2,y2)
    # d: step size

    total_dist = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    num_points = int(total_dist // d)

    # Handle cases with vertical (slope = inf) or horizontal (slope = 0)
    eps = 0.1
    if abs(y1 - y2) < eps:  # horizontal
        x_vals = np.linspace(start=x1, stop=x2, num=num_points)
        y_vals = np.ones(num_points) * y1
    elif abs(x1 - x2) < eps:  # vertical
        x_vals = np.ones(num_points) * x1
        y_vals = np.linspace(start=y1, stop=y2, num=num_points)
    else:
        slope = (y2 - y1) / (x2 - x1)
        dx = d / math.sqrt(1 + slope ** 2)
        dy = slope * dx

        x_vals = np.zeros(num_points)
        y_vals = np.zeros(num_points)

        # Start point
        x_vals[0] = x1
        y_vals[0] = y1

        # End point
        x_vals[num_points - 1] = x2
        y_vals[num_points - 1] = y2

        for i in range(1, num_points - 1):
            x_vals[i] = x_vals[i - 1] + dx
            y_vals[i] = y_vals[i - 1] + dy

    return merge(x_vals, y_vals)

# Matching sampled (x,y) point along branch to nearest available weather datapoint
def closest_point(point,points):
    min_index = cdist([point],points).argmin()
    return min_index, points[min_index]

# Loop through all branches and store the closest weather points - Only need to do this ONCE
# Same fixed points sampled throughout
num_branches = branches.shape[0]
branch_weather_points = {}
branch_weather_indices = {}

mac = False

if mac:
    file_path1 = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/weather_data_Jan2016.pkl'
else:
    file_path1 = "C:\\Users\\vinee\\OneDrive - Massachusetts Institute of Technology\\MIT\\Semesters\\Spring 2022\\15.S08\\Project\\weather_data_Jan2016.pkl"

if mac:
    file_path2 = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/lat_lon_data.pkl'
else:
    file_path2 = "C:\\Users\\vinee\\OneDrive - Massachusetts Institute of Technology\\MIT\\Semesters\\Spring 2022\\15.S08\\Project\\lat_lon_data.pkl"

with open(file_path1, 'rb') as file:
        x_orig_all = pickle.load(file)
        y_orig_all = pickle.load(file)
        temp_all = pickle.load(file)
        u_all = pickle.load(file)
        v_all = pickle.load(file)

with open(file_path2, 'rb') as file:
    a = pickle.load(file)
    b = pickle.load(file)
    c = pickle.load(file)
    d = pickle.load(file)
    e = pickle.load(file)
    f = pickle.load(file)
    lat_all = pickle.load(file)
    lon_all = pickle.load(file)

# Convert masked arrays to regular numpy arrays
lat_all = lat_all[0].compressed()
lon_all = lon_all[0].compressed()

lat_all_array = lat_all.ravel()
lon_all_array = lon_all.ravel()

# Convert lat-lon to utm projection
result = utm.from_latlon(lat_all_array,lon_all_array)
weather_x, weather_y = result[0], result[1]
weather_points = merge(weather_x, weather_y)

d = 1000  # units in km?

for i in range(num_branches):
    branch_sample_points = sample_points(branches.iloc[i].fromX, branches.iloc[i].fromY, branches.iloc[i].toX,
                                         branches.iloc[i].toY, d)
    branch_weather_indices[i] = [closest_point(pt, weather_points)[0] for pt in branch_sample_points]
    branch_weather_points[i] = [closest_point(pt, weather_points)[1] for pt in branch_sample_points]

#%% Get subset of valid points for which we have data
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

#%% Loop through all hours
# Load weather data
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
days_per_month = [31,29,31,30,31,30,31,31,30,31,30,31]
hours_per_month = days_per_month * 60 # h indexing
total_hours = sum(hours_per_month)

str1 = "C:\\Users\\vinee\\OneDrive - Massachusetts Institute of Technology\\MIT\\Semesters\\Spring 2022\\15.S08\\Project\\weather_data_"
str2 = "2016.pkl"

dlr_values = np.zeros(num_branches,total_hours)

for i in range(len(months)):
    file_name = str1 + months[i] + str2

    with open(file_name, 'wb') as file:
            x_orig_all = pickle.load(file)
            y_orig_all = pickle.load(file)
            temp_all = pickle.load(file)
            u_all = pickle.load(file)
            v_all = pickle.load(file)
            times = pickle.load(file)

    weather_points = merge(x_orig_all,y_orig_all)

    d = 100 # units in km?

    for i in range(num_branches):
        branch_sample_points = sample_points(branches.fromX,branches.fromY,branches.toX,branches.toY,d)
        weather_indices = [closest_point(pt,weather_points)[0] for pt in branch_sample_points]
        weather_points = [closest_point(pt,weather_points)[1] for pt in branch_sample_points]
        u = u_all