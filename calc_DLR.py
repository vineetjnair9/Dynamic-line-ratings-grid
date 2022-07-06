#%% IMPORTS
from cmath import sqrt
from statistics import mean
from powersimdata.input.grid import Grid
from datetime import datetime
from metpy.units import units
import pandas as pd
import numpy as np
import utm
import math
import pickle
import utm
import time

import requests
from bs4 import BeautifulSoup

from scipy.spatial.distance import cdist

#%%
personal = 1

if personal:
    branch_weather_points_path = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/branch_weather.pkl'
    available_days_path = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/available_days.pkl'
    str1 = "/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/weather_data_"
    results_path = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/dlr_vals_notcapped_conservative.pkl'
else: # NREL laptop
    branch_weather_points_path = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/branch_weather.pkl'
    available_days_path = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/available_days.pkl'
    str1 = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/weather_data_'
    results_path = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/dlr_vals_notcapped_conservative.pkl'

str2 = "2016.pkl"
grid = Grid(["Texas"])

buses = grid.bus

#  X and Y coordinates of buses
# Convert from WGS-84 (lat, lon) --> UTM projection
result = utm.from_latlon(buses.lat.to_numpy(),buses.lon.to_numpy())
buses['X'], buses['Y'] = result[0], result[1]

branches = grid.branch
num_branches = branches.shape[0]

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

def conductor_axis(branches,branch_num):
    fromY = branches['fromY'].iloc[branch_num]
    toY = branches['toY'].iloc[branch_num]
    fromX = branches['fromX'].iloc[branch_num]
    toX = branches['toX'].iloc[branch_num]

    if abs(fromX - toX) < 0.1:
        return math.pi/2 * np.sign(toY - fromY)
    else:
        return math.atan2(toY - fromY, toX - fromX)

def wind_dir(u,v):
    return math.atan2(v,u)

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

# Load weather data
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
days_per_month = [31,29,31,30,31,30,31,31,30,31,30,32]
hours_per_month = days_per_month * 24 # h indexing
total_hours = sum(hours_per_month)

#%% Loop through all branches and store the closest weather points - Only need to do this ONCE
# Same fixed points sampled throughout
branch_weather_points = {}
branch_weather_indices = {}

mac = True
if mac:
    file_path1 = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/weather_data_Jan2016.pkl'
else:
    file_path1 = "C:\\Users\\vinee\\OneDrive - Massachusetts Institute of Technology\\MIT\\Semesters\\Spring 2022\\15.S08\\DLR Project\\weather_data_Jan2016.pkl"

if mac:
    file_path2 = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/lat_lon_data.pkl'
else:
    file_path2 = "C:\\Users\\vinee\\OneDrive - Massachusetts Institute of Technology\\MIT\\Semesters\\Spring 2022\\15.S08\\DLR Project\\lat_lon_data.pkl"

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

d = 100  # units in km?

# Find nearest points
for i in range(num_branches):
    branch_sample_points = sample_points(branches.iloc[i].fromX, branches.iloc[i].fromY, branches.iloc[i].toX,
                                         branches.iloc[i].toY, d)
    branch_weather_indices[i] = [closest_point(pt, weather_points)[0] for pt in branch_sample_points]
    branch_weather_points[i] = [closest_point(pt, weather_points)[1] for pt in branch_sample_points]

#
with open('/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/branch_weather.pkl', 'wb') as file:
    pickle.dump(branch_weather_indices, file)
    pickle.dump(branch_weather_points, file)

# Find all hours with available data
hrs = {}
num_hours_per_date = {}
for date in pd.date_range(start='2016-1-1', end='2017-1-1', freq='D'):
    hrs[date] = avail_hours(date.year, date.month, date.day)
    num_hours_per_date[date] = len(hrs[date])
 
with open('/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/available_days.pkl', 'wb') as file:
    pickle.dump(hrs, file)
    pickle.dump(num_hours_per_date, file)

#%% Loop through all hours

with open(branch_weather_points_path, 'rb') as file:
        branch_weather_indices = pickle.load(file)
        branch_weather_points = pickle.load(file)

with open(available_days_path, 'rb') as file:
        hrs = pickle.load(file)
        num_hours_per_date = pickle.load(file)

# ERCOT TDSP ampacity assumptions
T_A_SLR_vals = [36, 25, 25,	40.55, 40, 40.55, 40, 25, 40, 25, 40, 43]
T_C_vals = [67.5, 67.5, 75, 93.33, 90, 67.5, 90, 67.5, 85, 67.5, 90, 75]

# DLR calculation parameters - ASSUMPTIONS
v_SLR = 0.6096 # [m/s]
T_A_SLR = mean(T_A_SLR_vals) # [C]
T_A_SLR = T_A_SLR + 273.15 # [K]
T_C_max = mean(T_C_vals) # [C]
T_C_max = 100 # [C] more conservative & >> T_A
T_C_max = T_C_max + 273.15 # [K]
# D = 25.4 * 1e-3 # mean diameter

alpha = v_SLR**(-0.26)
rho_f = 1.029  # Density of air (kg/m^3)
mu_f = 2.043 * 1e-5 # Dynamic viscosity of air [kg/m-s]
beta = (0.559/(v_SLR**0.26))*(rho_f/mu_f)**0.04
gamma = math.sqrt(1/(T_C_max - T_A_SLR))

branches['fromKV'] = buses.reindex(branches.from_bus_id)['baseKV'].values
branches['toKV']   = buses.reindex(branches.to_bus_id)['baseKV'].values

unique_weather_data = {}
num_unique_weather = np.zeros(num_branches)
branch_diameters = np.zeros(num_branches)
conductor_axes = np.zeros(num_branches)

# Preprocessing all branches
for i in range(num_branches):
    unique_weather_data[i] = list(set(branch_weather_indices[i]))
    num_unique_weather[i] = int(len(unique_weather_data[i]))
    current_rating = (branches['rateA'].iloc[i]*1000)/(math.sqrt(3)*branches['fromKV'].iloc[i])
    # Cap maximum conductor diameter at 1.88 in
    branch_diameters[i] = (min(0.001*current_rating + 0.2182,1.88)*0.0254) # [m]
    conductor_axes[i] = conductor_axis(branches,i)

#%%
dlr_values = np.ones((num_branches,total_hours))
dlr_values_temp = np.ones((num_branches,total_hours))
dlr_values_wind = np.ones((num_branches,total_hours))

hrs_for_each_date_indexed = list(hrs.values())

global_h = 0 # global hour counter
global_d = 0 # global day counter

for i in range(len(months)):
    print('Month: ', months[i])
    weather_data_file = str1 + months[i] + str2

    with open(weather_data_file, 'rb') as file:
        x_orig = pickle.load(file)
        y_orig = pickle.load(file)
        temp_all = pickle.load(file)
        u_all = pickle.load(file)
        v_all = pickle.load(file)
    
    month_h = 0

    for j in range(days_per_month[i]):
        print('Day: ',str(j))
        start = time.time()

        for k in range(24):

            if k in hrs_for_each_date_indexed[global_d]:

                temp_month = temp_all[month_h].ravel()
                u_month = u_all[month_h].ravel()
                v_month = v_all[month_h].ravel()

                for l in range(num_branches):

                    if branches.branch_device_type.iloc[l] == 'Transformer':
                        dlr_values[l,global_h] = 1.0
                        continue

                    num_unique = int(num_unique_weather[l])
                    dlrs = np.zeros(num_unique)
                    dlrs_wind = np.zeros(num_unique)
                    dlrs_temp = np.zeros(num_unique)

                    for m in range(num_unique):

                        n = unique_weather_data[l][m]
                        temp = temp_month[n].magnitude
                        u = u_month[n].magnitude
                        v = v_month[n].magnitude
                        speed = math.sqrt(u**2 + v**2)
                        phi = wind_dir(u,v) - conductor_axes[l]
                        K_angle = 1.194 - math.cos(phi) + 0.194*math.cos(2*phi) + 0.368*math.sin(2*phi)
                        eta_low_v = alpha * K_angle * speed**0.26
                        eta_high_v = beta * branch_diameters[l]**(0.04) * speed**0.3
                        eta_T = gamma*(T_C_max - temp)**0.5
                        # Not capped version: May be cases where DLR < SLR
                        dlrs[m] = max(eta_low_v*eta_T,eta_high_v*eta_T)
                        dlrs_wind[m] = max(eta_low_v,eta_high_v)
                        dlrs_temp[m] = eta_T
                        # Capped version
                        # dlrs[m] = max(1,eta_low_v*eta_T,eta_high_v*eta_T) 

                    dlr_values[l,global_h] = min(dlrs)
                    dlr_values_temp[l,global_h] = min(dlrs_temp)
                    dlr_values_wind[l,global_h] = min(dlrs_wind)

                month_h += 1

            global_h += 1

        global_d += 1
        end = time.time()
        print('Time: ', end - start)

#%%
with open(results_path, 'wb') as file:
    pickle.dump(dlr_values, file)
    pickle.dump(dlr_values_temp, file)
    pickle.dump(dlr_values_wind, file)

#%% Calculate lengths of all lines [in km]
# Filter out short & medium lines (<= 100km) - only apply DLR to those
branches['line_length'] = np.sqrt((branches.toX - branches.fromX) ** 2 + (branches.toY - branches.fromY) ** 2)/1000.0

short_med_branches = branches[branches['line_length'] <= 100.0]

# Find absolute branch indices corresponding to short/medium branch IDs
