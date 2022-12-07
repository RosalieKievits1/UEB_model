# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import Functions
import SVF

#plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
# csfont = {'fontname':'Arial (sans-serif)'}
# hfont = {'fontname':'Arial (sans-serif)'}

#
# nr_steps = Functions.nr_steps
# T_air = Functions.T_air

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
[T_r, T_w,T_road, LW_net_roof, SW_net_roof, LHF_roof, SHF_roof, G_out_surf_roof] = Functions.HeatEvolution(SVF.m5_data,Constants.nr_of_steps,Constants.timestep,180,40, Functions.T_2m)
Functions.PlotSurfaceTemp(T_r,T_w,T_road,Constants.nr_of_steps)

time = (np.arange(Constants.nr_of_steps))# * Constants.timestep)/3600
#
plt.figure()
plt.plot(time,LW_net_roof, label="LW")
plt.plot(time,SW_net_roof, label="SW net roof")
plt.plot(time,Functions.SW_down,label="SW down")
plt.plot(time,LHF_roof, label="LHF")
plt.plot(time,SHF_roof, label="SHF")
plt.plot(time,G_out_surf_roof, label="G")
#
plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
plt.xlabel("Time [h]")
plt.ylabel("SW down and SW net")
plt.legend()
plt.show()
