#%%
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from powersimdata.input.grid import Grid
import matplotlib.pyplot as plt

personal = 0

if personal:
    with open('/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/dlr_vals_notcapped.pkl', 'rb') as file:
        dlr_values = pickle.load(file)
        dlr_values_temp = pickle.load(file)
        dlr_values_wind = pickle.load(file)
else:
    with open('/Users/vjagadee/OneDrive - Massachusetts Institute of Technology/MIT/Semesters/Spring 2022/15.S08/DLR Project/dlr_vals_notcapped.pkl', 'rb') as file:
        dlr_values = pickle.load(file)
        dlr_values_temp = pickle.load(file)
        dlr_values_wind = pickle.load(file)

grid = Grid(["Texas"])
branches = grid.branch
num_branches = branches.shape[0]
num_hours = dlr_values.shape[1]

#%%
X = range(1,num_branches+1)
Y = range(1,num_hours+1)

XX, YY = np.meshgrid(X, Y)
XX = XX.T
YY = YY.T

#%% Compare DLR (both wind velocity + temp) vs AAR (temp only) vs wind velocity only
# Analyze DLR factors over all branches and for all hours
# Critical value = 1
plt.contour(XX, YY, dlr_values, cmap='coolwarm');
plt.xlabel('Branch number')
plt.ylabel('Hour')
plt.title('DLR factors using both wind velocity and temperature')
plt.savefig('DLR_dist_all.png',dpi=600)
plt.colorbar()
#TODO: try plotting with higher DPI/resolution

#%% Distribution of DLR factors using temp only
plt.contour(XX, YY, dlr_values_temp, cmap='coolwarm');
plt.xlabel('Branch number')
plt.ylabel('Hour')
plt.title('DLR factors using only ambient air temperature (AAR)')
plt.savefig('DLR_dist_temp_all.png',dpi=600)

#%% Distribution of DLR factors using wind speed only
plt.contour(XX, YY, dlr_values_wind, cmap='coolwarm');
plt.xlabel('Branch number')
plt.ylabel('Hour')
plt.title('DLR factors using only wind speed/direction')
plt.savefig('DLR_dist_wind_all.png',dpi=600)

#%% Average DLR factors for each branch averaged over entire year
dlr_branch_avg = np.mean(dlr_values,axis=1)
dlr_temp_branch_avg = np.mean(dlr_values_temp,axis=1)
dlr_wind_branch_avg = np.mean(dlr_values_wind,axis=1)

#%% Check hours/branches when DLR < SLR
dlr_less = dlr_values < 1.0
# % of hours and branches for which DLR falls below SLR
dlr_less_fraction = (np.count_nonzero(dlr_less)/(num_branches*num_hours)) * 100

# Find indices (hour + branch) for which DLR < SLR
dlr_less_indices = np.nonzero(dlr_less)
dlr_less_branches = dlr_less_indices[0]
dlr_less_hours = dlr_less_indices[1]

# Summary statistics
dlr_median_val = np.median(dlr_values)
dlr_mean_val = np.mean(dlr_values)
dlr_min_val = np.min(dlr_values)
dlr_max_val = np.max(dlr_values)

# %%
