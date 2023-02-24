#%% IMPORTS
from cmath import sqrt
from statistics import mean
from matplotlib.font_manager import json_dump
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
import json
import jsonpickle
import angles
import requests
from bs4 import BeautifulSoup
from scipy.spatial.distance import cdist

grid = Grid(["Texas"])

# grid_json = jsonpickle.encode(grid,keys=True)
# json.dump(grid, open('/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/grid.json', 'w'))

personal = 1

str2 = "2016.pkl"

if personal:
    branch_weather_points_path = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/branch_weather.pkl'
    available_days_path = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/available_days.pkl'
    str1 = "/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/weather_data_"
    results_path = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/dlr_vals_final_'
else: # NREL laptop
    branch_weather_points_path = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/branch_weather.pkl'
    available_days_path = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/available_days.pkl'
    str1 = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/weather_data_'
    results_path = '/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/dlr_vals_final_'

buses = grid.bus
branches = grid.branch

#  X and Y coordinates of buses
# Convert from WGS-84 (lat, lon) --> UTM projection
result = utm.from_latlon(buses.lat.to_numpy(),buses.lon.to_numpy())
buses['X'], buses['Y'] = result[0], result[1]

num_branches = branches.shape[0]

# Assign start and end X and Y coordinates
from_result = utm.from_latlon(branches.from_lat.to_numpy(),branches.from_lon.to_numpy())
branches['fromX'], branches['fromY'] = from_result[0], from_result[1]

to_result = utm.from_latlon(branches.to_lat.to_numpy(),branches.to_lon.to_numpy())
branches['toX'], branches['toY'] = to_result[0], to_result[1]

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

def K_angle(phi):
    phi = angles.normalize(phi,-math.pi/2.0, math.pi/2.0)
    return 1.194 - math.cos(phi) + 0.194*math.cos(2*phi) + 0.368*math.sin(2*phi)

# Load weather data
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
days_per_month = [31,29,31,30,31,30,31,31,30,31,30,31]
hours_per_month = days_per_month * 24 # h indexing
total_hours = sum(hours_per_month)

# Loop through all hours
with open(branch_weather_points_path, 'rb') as file:
        branch_weather_indices = pickle.load(file)
        branch_weather_points = pickle.load(file)

with open(available_days_path, 'rb') as file:
        hrs = pickle.load(file)
        num_hours_per_date = pickle.load(file)

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

# ERCOT TDSP ampacity assumptions
T_A_SLR_vals = [36, 25, 25,	40.55, 40, 40.55, 40, 25, 40, 25, 40, 43]
T_C_vals = [67.5, 67.5, 75, 93.33, 90, 67.5, 90, 67.5, 85, 67.5, 90, 75]

# DLR calculation parameters - ASSUMPTIONS
v_SLR = 0.6096 # [m/s]
T_A_SLR = mean(T_A_SLR_vals) # [C]

T_A_SLR = T_A_SLR + 273.15 # [K]
alpha = v_SLR**(-0.26)
rho_f = 1.029  # Density of air (kg/m^3)
mu_f = 2.043 * 1e-5 # Dynamic viscosity of air [kg/m-s]
beta = (0.566/(v_SLR**0.26))*(rho_f/mu_f)**0.04

T_C_max_vals = [round(mean(T_C_vals)), 100.0, 110.0]
K_SLR_vals = [K_angle(0.0), K_angle(math.pi/4.0), K_angle(math.pi/2.0)]
SLR_wind_angle = [0, 45, 90]

#%% Loop over all cases

for TC_case in range(len(T_C_max_vals)):
    for KSLR_case in range(len(K_SLR_vals)):

        T_C_max = T_C_max_vals[TC_case] # [C]
        T_C_max_str = str(T_C_max)

        K_SLR = K_SLR_vals[KSLR_case]
        SLR_wind_angle_str = str(SLR_wind_angle[KSLR_case])

        T_C_max = T_C_max + 273.15 # [K]
        gamma = math.sqrt(1/(T_C_max - T_A_SLR))

        dlr_values = np.ones((num_branches,total_hours))
        dlr_values_temp = np.ones((num_branches,total_hours))
        dlr_values_wind = np.ones((num_branches,total_hours))

        hrs_for_each_date_indexed = list(hrs.values())

        global_h = 0 # global hour counter
        global_d = 0 # global day counter

        for i in range(len(months)):
            # print('Month: ', months[i])
            weather_data_file = str1 + months[i] + str2

            with open(weather_data_file, 'rb') as file:
                x_orig = pickle.load(file)
                y_orig = pickle.load(file)
                temp_all = pickle.load(file)
                u_all = pickle.load(file)
                v_all = pickle.load(file)
            
            month_h = 0

            for j in range(days_per_month[i]):
                # print('Day: ',str(j))
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
                                eta_low_v = alpha * math.sqrt(K_angle(phi)/K_SLR) * speed**0.26
                                eta_high_v = beta * math.sqrt(K_angle(phi)/K_SLR) * branch_diameters[l]**(0.04) * speed**0.3
                                eta_T = gamma*(T_C_max - temp)**0.5
                                # Not capped version: May be cases where DLR < SLR
                                dlrs[m] = max(eta_low_v*eta_T,eta_high_v*eta_T)
                                dlrs_wind[m] = max(eta_low_v,eta_high_v)
                                dlrs_temp[m] = eta_T 

                            dlr_values[l,global_h] = min(dlrs)
                            dlr_values_temp[l,global_h] = min(dlrs_temp)
                            dlr_values_wind[l,global_h] = min(dlrs_wind)

                        month_h += 1

                    global_h += 1

                global_d += 1
                end = time.time()
                # print('Time: ', end - start)

        with open(results_path + 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str, 'wb') as file:
            pickle.dump(dlr_values, file)
            pickle.dump(dlr_values_temp, file)
            pickle.dump(dlr_values_wind, file)
