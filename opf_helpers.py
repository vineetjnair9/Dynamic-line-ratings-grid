import numpy as np
import pandas as pd

def calc_Bmatrix(buses,branches):
    N = buses.shape[0] # no. of buses
    B = np.zeros((N,N))
    for i in range(N): # Loop through each bus
        busID = buses.index[i]
        bus_branches = branches[(branches.from_bus_id==busID) | (branches.to_bus_id==busID)]
        to_
        for j in range(N):
            B[i,j]