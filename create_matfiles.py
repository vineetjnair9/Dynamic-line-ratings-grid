import numpy as np
import pandas as pd
from powersimdata import Grid
import sys

my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

fnames = list(pd.date_range('2016-1-1 0:00', '2016-12-31 23:00', freq='H'))
####fnames = list(pd.date_range('2016-1-31 10:00', '2016-1-31 11:00', freq='H'))

my_fnames = fnames[my_task_id:len(fnames):num_tasks]

griddata = ''
texas = Grid(["Texas"])

# Pre-clean the dataset
# [warn | PowerModels]: angmin and angmax values are 0, widening these values on branch 1459 to +/- 60.0 deg.
texas.branch['angmin'] = -60
texas.branch['angmax'] = +60

# Set all base MVA to 100
# R,X,B values should be MVA 100 by default:
# https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Power_Flow_Solution_General.htm
texas.plant['mBase'] = 100

# Need to update nodal demand and gens
demand = pd.read_csv(griddata + 'demand.csv', index_col=0, parse_dates=True)
demand.columns = demand.columns.astype(int)
"""
def cleaner(s):
    df  = pd.read_csv(griddata + s + '.csv', index_col=0, parse_dates=True)
    df.columns = df.columns.astype(int)
    df = df.loc[:, df.columns.isin(texas.plant.index)]
    df.to_csv(griddata + s + '_clean.csv')
cleaner('hydro')
cleaner('wind')
cleaner('solar')
"""
hydro  = pd.read_csv(griddata + 'hydro_clean.csv', index_col=0, parse_dates=True)
solar = pd.read_csv(griddata + 'solar_clean.csv', index_col=0, parse_dates=True)
wind = pd.read_csv(griddata + 'wind_clean.csv', index_col=0, parse_dates=True)
gen_t = pd.concat([wind, solar, hydro], axis=1)  # Combo of available renewables
gen_t.columns = gen_t.columns.astype(int)


# Load distribution factors
LOAD_DIST = texas.bus['Pd'] / texas.bus.groupby('zone_id')['Pd'].sum().reindex(texas.bus['zone_id']).values

# Temp object to iterate over times
bus_t = texas.bus.copy()
plant_t = texas.plant.copy()
branch_t = texas.branch.copy()

# Convert all active gen buses to type 2 (PV)
bus_t.loc[bus_t.index.isin(plant_t['bus_id']), 'type'] = 2
bus_t.loc[texas.bus[texas.bus['type']==3].index[0], 'type'] = 3 # Keep original reference bus

# Clean plant status
plant_t['Pmin'] = plant_t['Pmin']*plant_t['status'] # if not online, Pmin = Pmax = 0
plant_t.loc[plant_t['type'].isin(['ng','coal']), 'Pmin'] = 0 # set fossil units to be flexible
plant_t['Pmax'] = plant_t['Pmax']*plant_t['status'] # if not online, Pmin = Pmax = 0
plant_t['status'] = 1 # Then set all status = 1 to simplify output

plant_t['Vg'] = texas.bus.reindex(texas.plant.bus_id)['Vm'].values # set voltages equal
# Reverse direction to match the parallel line
tmp = branch_t.loc[100954, 'from_bus_id']
branch_t.loc[100954, 'from_bus_id'] = branch_t.loc[100954, 'to_bus_id']
branch_t.loc[100954, 'to_bus_id'] = tmp
# Clean the gencost table
gcc = texas.gencost['after'].copy()
gcc.loc[gcc[['c0','c1','c2']].sum(1)==0, 'n'] = 0   # set 0 cost plants to have no cost


# Convert to Matpower format, and save to .m file
# https://matpower.org/docs/ref/matpower5.0/caseformat.html
def write_hour_case(dt):
    # Updated demand in hour
    bus_t['Pd'] = demand.loc[dt].reindex(texas.bus['zone_id']).values * LOAD_DIST.values
    new_gens = gen_t.loc[dt].reindex(plant_t.index)  # Update the renewable gens
    plant_t['Pmax'] = new_gens.combine_first(texas.plant['Pmax'])  # New time series takes priority

    OUTPUT = 'input_cases/'
    OUTPUT_NAME = 'texas_' + dt.strftime('%Y-%m-%d_%H.m')

    bus = 'mpc.bus = [\n' + \
          bus_t.sort_index().reset_index().iloc[:, :13].to_string(header=False, index=False).replace('\n',';\n') + \
          '\n];\n'

    # Drop plant ID
    gen = 'mpc.gen = [\n' + \
          plant_t.sort_index().iloc[:,:11].to_string(header=False, index=False).replace('\n',';\n') + \
          '\n];\n'

    gencost = 'mpc.gencost = [\n' + \
              gcc.sort_index().iloc[:,:-1].to_string(header=False, index=False).replace('\n',';\n') + \
              '\n];\n'

    # Need to include angmin, angmax
    branch = 'mpc.branch = [\n' + \
             branch_t.iloc[:,:13].to_string(header=False, index=False).replace('\n',';\n') + \
             '\n];\n'

    text_file = open(OUTPUT + OUTPUT_NAME, "w")

    n = text_file.write('function mpc = case3\n' + 'mpc.baseMVA = 100.0;\n' + "mpc.version = '2';\n" + bus + gen + gencost + branch)
    text_file.close()


for dt in my_fnames:
    print(dt)
    write_hour_case(dt=dt)
