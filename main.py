# import packages
import numpy as np
import pandas as pd
import Constants
import SVF
import matplotlib.pyplot as plt
import Functions

# This script is for simulating an urban energy balance, taking into consideration the town geometry.
# 1: first we define all governing equations and test the surface energy and temperature for a flat surface
# 2: Next we implement a surface with height differences (start with infinite canyon)
# 3: Implement solar angle and shading effects
# 4: implement reflection effects
# 5: implement snow and rain surface obstruction
# 6: implement human heat sources

nr_steps = Functions.nr_steps
T_air = Functions.T_air

"""MASSON"""
"""Plotting of the layers temperatures for each material"""
#[map_temperatures_roof,map_temperatures_wall,map_temperatures_road] = Functions.Masson_model(Constants.T_building,Constants.T_ground,Constants.T_air,nr_steps,Constants.H_W)
#Functions.plotTemp_Masson(map_temperatures_roof)
#Functions.plotTemp_Masson(map_temperatures_road)
#Functions.plotTemp_Masson(map_temperatures_wall)
#Functions.plotTempComparison_Masson(map_temperatures_roof,map_temperatures_wall,map_temperatures_road,1)

"""City using SVF and shadow casting"""
data = SVF.datasquare(SVF.dtm1,SVF.dsm1,SVF.dtm2,SVF.dsm2,SVF.dtm3,SVF.dsm3,SVF.dtm4,SVF.dsm4)
#print(SVF.geometricProperties(data,SVF.gridboxsize))
blocklength = int(data.shape[0]/2*data.shape[1]/2)
coords = SVF.coordheight(data)
print(SVF.reshape_SVF(data,coords,294,48,52,10,reshape=False))

# print("These are the Sky View Factors")
# print(SVF_matrix)
# print("These are the Shadowfactors")
# print(SF_matrix)
# print("These are the average temperatures")
# print(Functions.HeatEvolution(data,nr_steps,Constants.timestep,SVF_matrix,SF_matrix))
#
