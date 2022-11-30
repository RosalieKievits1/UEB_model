# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import Functions
import SVF
import Sunpos

#plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
# csfont = {'fontname':'Arial (sans-serif)'}
# hfont = {'fontname':'Arial (sans-serif)'}

# plt.figure()
# x = np.linspace(0,100,100)
# y = x**2
# plt.title('The built area fraction')
# plt.xlabel('xlabel')
# plt.show()
#
# nr_steps = Functions.nr_steps
# T_air = Functions.T_air

"""MASSON"""
"""Plotting of the layers temperatures for each material"""
#[map_temperatures_roof,map_temperatures_wall,map_temperatures_road] = Functions.Masson_model(Constants.T_building,Constants.T_ground,Constants.T_air,nr_steps,Constants.H_W)
#Functions.plotTemp_Masson(map_temperatures_roof)
#Functions.plotTemp_Masson(map_temperatures_road)
#Functions.plotTemp_Masson(map_temperatures_wall)
#Functions.plotTempComparison_Masson(map_temperatures_roof,map_temperatures_wall,map_temperatures_road,1)

"""City using SVF and shadow casting"""
# #print(SVF.geometricProperties(data,SVF.gridboxsize))
# blocklength = int(data.shape[0]/2*data.shape[1]/2)
# coords = SVF.coordheight(data)
# print(SVF.reshape_SVF(data,coords,294,48,52,10,reshape=False))

# print("These are the Sky View Factors")
# print(SVF_matrix)
# print("These are the Shadowfactors")
# print(SF_matrix)
# print("These are the average temperatures")
# print(Functions.HeatEvolution(data,nr_steps,Constants.timestep,azimuth,zenith))

"""Calculate the Shadow casting for 24 hours on one day"""
# data = SVF.readdata(SVF.minheight,SVF.dsm1,SVF.dtm1)
# hour = np.linspace(0,24,25)
# for t in len(hour):
#     "Calculate the azimuth and zenith for every hour"
#     [zenith,azimuth] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,hour[t],radians=True)
#
#     "Calculate average surface temperatures for roof and road surface types"
#     [t_roof_ave, t_road_ave, t_ave] = Functions.HeatEvolution(data,Constants.nr_of_steps,Constants.timestep,azimuth,zenith)
#
# Functions.PlotGreyMap(SVF.data,middle=False,v_max=1)
[T_r, T_w,T_road] = Functions.HeatEvolution(SVF.data,Constants.nr_of_steps,Constants.timestep,180,40, Functions.T_2m,SVF.SVF_matrix)
Functions.PlotSurfaceTemp(T_r, T_w,T_road,Constants.nr_of_steps)
