#%%
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from powersimdata.input.grid import Grid
import matplotlib.pyplot as plt
import angles
import math
from pathlib import Path
from statistics import mean
import matplotlib.colors
from scipy.interpolate import make_interp_spline
import utm
from scipy.ndimage.filters import gaussian_filter1d

def K_angle(phi):
    phi = angles.normalize(phi,-math.pi/2.0, math.pi/2.0)
    return 1.194 - math.cos(phi) + 0.194*math.cos(2*phi) + 0.368*math.sin(2*phi)

personal = 1

grid = Grid(["Texas"])
branches = grid.branch
num_branches = branches.shape[0]

from_result = utm.from_latlon(branches.from_lat.to_numpy(),branches.from_lon.to_numpy())
branches['fromX'], branches['fromY'] = from_result[0], from_result[1]

to_result = utm.from_latlon(branches.to_lat.to_numpy(),branches.to_lon.to_numpy())
branches['toX'], branches['toY'] = to_result[0], to_result[1]

plots_path = Path('/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/Plots')

with open('/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/available_days.pkl', 'rb') as file:
        hrs = pickle.load(file)
        num_hours_per_date = pickle.load(file)

keys = list(hrs)
hr_indices_with_available_data = []
for day in range(len(keys)):
    hr_indices_with_available_data.extend([h+24*day for h in hrs[keys[day]]])

T_C_vals = [67.5, 67.5, 75, 93.33, 90, 67.5, 90, 67.5, 85, 67.5, 90, 75]
T_C_max_vals = [round(mean(T_C_vals)), 100.0, 110.0]
K_SLR_vals = [K_angle(0.0), K_angle(math.pi/4.0), K_angle(math.pi/2.0)]
SLR_wind_angle = [0, 45, 90]

#%%
TC_case = 0
KSLR_case = 0 

T_C_max = T_C_max_vals[TC_case] # [C]
T_C_max_str = str(T_C_max)

K_SLR = K_SLR_vals[KSLR_case]
SLR_wind_angle_str = str(SLR_wind_angle[KSLR_case])

results_path = '/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/DLR Project/dlr_vals_final_'

with open(results_path + 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str, 'rb') as file:
    dlr_values = pickle.load(file)
    dlr_values_temp = pickle.load(file)
    dlr_values_wind = pickle.load(file)

num_hours = dlr_values.shape[1]

# Filter out DLR values with missing data
dlr_values_filter = dlr_values[:,hr_indices_with_available_data]
dlr_values_temp_filter = dlr_values_temp[:,hr_indices_with_available_data]
dlr_values_wind_filter = dlr_values_wind[:,hr_indices_with_available_data]

# Calculate lengths of all lines [in km]
# Filter out short & medium lines (<= 100km) - only apply DLR to those
branches['line_length'] = np.sqrt((branches.toX - branches.fromX) ** 2 + (branches.toY - branches.fromY) ** 2)/1000.0

short_med_branches = branches[branches['line_length'] <= 100.0]
long_branches = branches[branches['line_length'] > 100.0]

# Find absolute branch indices corresponding to short/medium branch IDs
short_med_branch_indices = np.where(branches['line_length'] <= 100.0)[0]

dlr_values_filter_shortmed = dlr_values_filter[short_med_branch_indices,:]
dlr_values_temp_filter_shortmed = dlr_values_temp_filter[short_med_branch_indices,:]
dlr_values_wind_filter_shortmed = dlr_values_wind_filter[short_med_branch_indices,:]

#%%
# X = range(short_med_branches.shape[0]) # range(num_branches)
X = short_med_branch_indices
Y = hr_indices_with_available_data

XX, YY = np.meshgrid(X, Y)
XX = XX.T
YY = YY.T

#%% Compare DLR (both wind velocity + temp) vs AAR (temp only) vs wind velocity only
# Analyze DLR factors over all branches and for all hours
# Critical value = 1
fig, ax = plt.subplots()
cs = plt.contourf(XX, YY, dlr_values_filter_shortmed, cmap='coolwarm');
norm= matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=cs.levels)
plt.xlabel('Branch number')
plt.ylabel('Hour')
plt.title('DLR factors using both wind velocity and temperature')
case_name = 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str
img_name = '_DLR_dist_all.png'
filename = case_name + img_name
plt.savefig(plots_path / filename,dpi=600)
plt.show()

#%% Distribution of DLR factors using temp only
fig, ax = plt.subplots()
cs = plt.contourf(XX, YY, dlr_values_temp_filter_shortmed, cmap='coolwarm');
norm= matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=cs.levels)
plt.xlabel('Branch number')
plt.ylabel('Hour')
plt.title('DLR factors using only ambient air temperature (AAR)')
case_name = 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str
img_name = '_DLR_dist_temp.png'
filename = case_name + img_name
plt.savefig(plots_path / filename,dpi=600)
plt.show()

#%% Distribution of DLR factors using wind speed only
fig, ax = plt.subplots()
cs = plt.contourf(XX, YY, dlr_values_wind_filter_shortmed, cmap='coolwarm');
norm= matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=cs.levels)
plt.xlabel('Branch number')
plt.ylabel('Hour')
plt.title('DLR factors using only wind speed/direction')
case_name = 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str
img_name = '_DLR_dist_wind.png'
filename = case_name + img_name
plt.savefig(plots_path / filename,dpi=600)
plt.show()

#%% Average DLR factors for each branch averaged over entire year
dlr_branch_avg = np.mean(dlr_values_filter_shortmed,axis=1)
dlr_temp_branch_avg = np.mean(dlr_values_temp_filter_shortmed,axis=1)
dlr_wind_branch_avg = np.mean(dlr_values_wind_filter_shortmed,axis=1)

#%% DLR factor averaged across all branches for entire year
dlr_avg = np.mean(dlr_values_filter_shortmed,axis=0)
dlr_temp_avg = np.mean(dlr_values_temp_filter_shortmed,axis=0)
dlr_wind_avg = np.mean(dlr_values_wind_filter_shortmed,axis=0)

#%% 
fig, ax = plt.subplots()
ax.plot(hr_indices_with_available_data,dlr_avg,label='Both v and T')
ax.plot(hr_indices_with_available_data,dlr_temp_avg,label='T only (AAR)')
ax.plot(hr_indices_with_available_data,dlr_wind_avg,label='v only')
ax.legend()
plt.xlabel('Hour')
plt.ylabel('DLR factor averaged across all branches')
case_name = 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str
img_name = '_DLR_avg_overtime.png'
filename = case_name + img_name
plt.savefig(plots_path / filename,dpi=600)

#%% Check hours/branches when DLR < SLR
dlr_less = dlr_values_filter_shortmed < 1.0
total_points = dlr_values_filter_shortmed.shape[0] * dlr_values_filter_shortmed.shape[1]
# % of hours and branches for which DLR falls below SLR
dlr_less_fraction = (np.count_nonzero(dlr_less)/(total_points)) * 100

# Find indices (hour + branch) for which DLR < SLR
dlr_less_indices = np.nonzero(dlr_less)
dlr_less_branches = dlr_less_indices[0]
dlr_less_hours = dlr_less_indices[1]

# Summary statistics
dlr_median_val = np.median(dlr_values)
dlr_mean_val = np.mean(dlr_values)
dlr_min_val = np.min(dlr_values)
dlr_max_val = np.max(dlr_values)

#%% Total capacity of all branches over time 
# Include long lines in total capacity but don't apply DLR to them
ratings = short_med_branches['rateA'].to_numpy()
Total_capacity_slr = np.sum(branches['rateA'])*np.ones((len(hr_indices_with_available_data),1))
Total_capacity_dlr = np.dot(dlr_values_filter_shortmed.T,ratings) + np.sum(long_branches['rateA'])
Total_capacity_dlr_temp = np.dot(dlr_values_temp_filter_shortmed.T,ratings) + np.sum(long_branches['rateA'])
Total_capacity_dlr_wind = np.dot(dlr_values_wind_filter_shortmed.T,ratings) + np.sum(long_branches['rateA'])

fig, ax = plt.subplots()
ax.plot(hr_indices_with_available_data,Total_capacity_slr,label='SLR')
ax.plot(hr_indices_with_available_data,Total_capacity_dlr,label='DLR with both v and T')
ax.plot(hr_indices_with_available_data,Total_capacity_dlr_temp,label='DLR with T only (AAR)')
ax.plot(hr_indices_with_available_data,Total_capacity_dlr_wind,label='DLR with v only')
ax.legend()
plt.xlabel('Hour')
plt.ylabel('Total capacity of all branches')
case_name = 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str
img_name = '_DLR_total_overtime.png'
filename = case_name + img_name
plt.savefig(plots_path / filename,dpi=600)

# %%
# Smoothing data
sigma = 30
Total_capacity_dlr_smoothed = gaussian_filter1d(Total_capacity_dlr,sigma=sigma)
Total_capacity_dlr_temp_smoothed = gaussian_filter1d(Total_capacity_dlr_temp,sigma=sigma)
Total_capacity_dlr_wind_smoothed = gaussian_filter1d(Total_capacity_dlr_wind,sigma=sigma)

fig, ax = plt.subplots()
ax.plot(hr_indices_with_available_data,Total_capacity_slr,label='SLR')
ax.plot(hr_indices_with_available_data,Total_capacity_dlr_smoothed,label='DLR')
ax.plot(hr_indices_with_available_data,Total_capacity_dlr_temp_smoothed,label='T only')
ax.plot(hr_indices_with_available_data,Total_capacity_dlr_wind_smoothed,label='v only')
plt.xlabel('Hour')
plt.ylabel('Total capacity of all branches')
case_name = 'TC_' + T_C_max_str + '_SLRphi_' + SLR_wind_angle_str
img_name = '_DLR_total_overtime_smoothed.png'
filename = case_name + img_name
plt.tight_layout()
ax.legend(loc='best')
plt.savefig(plots_path / filename,dpi=600)
plt.show()

# %% Distributions of DLR factors - PDF over all branches (averaged over the whole yr)
sns.displot(dlr_branch_avg,kind='kde',bw_adjust=1)

# %% Distributions of DLR factors - CDF over all branches (averaged over the whole yr)
sns.displot(dlr_branch_avg,kind='ecdf')

# %% Distributions of DLR factors - PDF over the year (averaged over all branches)
sns.displot(dlr_avg,kind='kde',bw_adjust=0.5)

# %% Distributions of DLR factors - PDF over the year (averaged over all branches)
sns.displot(dlr_avg,kind='ecdf')
