# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import Functions
#import SVF

#plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
# csfont = {'fontname':'Arial (sans-serif)'}
# hfont = {'fontname':'Arial (sans-serif)'}
plt.close('all')
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

time = (np.arange(Constants.nr_of_steps)) #in hours
#plt.plot(time,Functions.T_2m)
# delta_d = 0.03
# T_soil_num_grass = Functions.NumericalSoil(time,Constants.timestep,delta_d,Constants.lamb_grass,Constants.C_grass,Constants.layers,Functions.T_2m)
# T_soil_num_asp = Functions.NumericalSoil(time,Constants.timestep,delta_d,Constants.lamb_asphalt,Constants.C_asphalt,Constants.layers,Functions.T_2m)
# style = ['solid',(0,(1,1)),(0,(1,2)),(0,(1,4))]
# plt.figure()
# for l in range(Constants.layers):
#     if l%5==0:
#         stl_idx = int(l/5)
#         plt.plot(time/6,T_soil_num_grass[:,l],'green',linestyle=style[stl_idx],label='Grass at depth ' + str(np.around(l*delta_d,2)) + 'm')
#         plt.plot(time/6,T_soil_num_asp[:,l],'black',linestyle=style[stl_idx],label='Asphalt at depth ' + str(np.around(l*delta_d,2)) + 'm')
# #plt.legend()
# plt.xlabel("time [h]")
# plt.ylabel("Temperature [K]")
# plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
# plt.show()
# thickness = np.linspace(0,1,50)
# T_soil = np.empty((len(thickness)))
# Num_soil = np.empty((len(thickness)))
# plt.figure()
# for d in range(len(thickness)):
#     T_soil[d] = Functions.AnalyticalSoil(0,thickness[d],Constants.lamb_grass,Constants.C_grass)
#     #plt.plot(T_soil_num[d],thickness)
#     Num_soil[d] = T_soil_num[0,d]
# #plt.plot(Num_soil,thickness)
# plt.plot(T_soil,thickness,'-')
# plt.ylim(max(thickness), min(thickness))
# plt.show()

# plt.plot(time,Functions.T_2m[:Constants.nr_of_steps], label="Forcing Temp")
# d = np.sum(Constants.d_roof[:])
# T_soil = Functions.AnalyticalSoil(time,d,Constants.lamb_grass,Constants.C_grass)
# plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
# plt.xlabel("Time [h]")
# plt.ylabel("Temperature for layer [K]")
# plt.legend(loc='upper right')
# plt.show()

# time = (np.arange(Constants.nr_of_steps))# * Constants.timestep/3600)

[T_roof, T_wall,T_road, LW_net_roof, SW_net_roof, LHF_roof, SHF_roof, G_out_surf_roof] = Functions.HeatEvolution(Constants.nr_of_steps,Constants.timestep)
Functions.PlotSurfaceTemp(T_roof,T_wall,T_road,Constants.nr_of_steps)
Functions.PlotTempLayers(T_wall,Constants.nr_of_steps)
Functions.PlotSurfaceFluxes(LW_net_roof, SW_net_roof, G_out_surf_roof, LHF_roof, SHF_roof)
# Functions.PlotSurfaceTemp(T_roof_g,T_wall_g,T_road_g,Constants.nr_of_steps)
# Functions.PlotTempLayers(T_roof_g,Constants.nr_of_steps)
# Functions.PlotSurfaceFluxes(LW_net_roof_g, SW_net_roof_g, G_out_surf_roof_g, LHF_roof_g, SHF_roof_g)
