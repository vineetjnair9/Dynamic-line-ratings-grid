import numpy as np
import pandas as pd

def calc_Bmatrix(buses,branches):
    N = buses.shape[0] # no. of buses
    B = np.zeros((N,N))
    for i in range(N): # Loop through each bus
        busID = buses.index[i]
        bus_branches = branches[(branches.from_bus_id==busID) | (branches.to_bus_id==busID)]
        to_buses = set(bus_branches.from_bus_id) | set(bus_branches.to_bus_id)
        to_buses.discard(busID)
        to_buses = list(to_buses)
        m = buses.index.get_loc(busID)
        b_vals = bus_branches.b
        B[m,m] = sum(b_vals)
        for j in range(len(to_buses)):
            n = buses.index.get_loc(to_buses[j])
            other_bus = to_buses[j]
            curr_bus_branches = bus_branches[(bus_branches.from_bus_id==other_bus) | (bus_branches.to_bus_id==other_bus)]
            B[m,n], B[n,m] = -sum(curr_bus_branches.b)
    
    return B