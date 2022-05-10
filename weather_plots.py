#%%
from powersimdata.input.grid import Grid
from datetime import datetime
from metpy.units import units
import pandas as pd
import numpy as np
import pickle
import utm
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import imageio
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import mapclassify as mc

grid = Grid(["Texas"])

buses = grid.bus

#  X and Y coordinates of buses
# Convert from WGS-84 (lat, lon) --> UTM projection
result = utm.from_latlon(buses.lat.to_numpy(),buses.lon.to_numpy())
buses['X'], buses['Y'] = result[0], result[1]

branches = grid.branch
num_branches = branches.shape[0]

with open('/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/weather_data_Jan2016.pkl', 'rb') as file:
    x_orig = pickle.load(file)
    y_orig = pickle.load(file)
    temp_all = pickle.load(file)
    u_all = pickle.load(file)
    v_all = pickle.load(file)
# %% Just select data for 1st hour to plot 1 snapshot
x = x_orig[0]
y = y_orig[0]
temp = temp_all[0]
u = u_all[0]
v = v_all[0]

# %%
