# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import Functions
import SVF
import Sunpos
import pickle

"Data import"
data = SVF.data
data_water = SVF.data_water
gridratio = 125
gr_SVF = 1
gridboxsize = SVF.gridboxsize

"Azimuth and zenith angle based on the day of the year"
days = 2
nr_of_steps = int(24*3600/Constants.timestep*days)
startday = Constants.julianday

Azi = np.empty((nr_of_steps))
El = np.empty((nr_of_steps))
Zenith = np.empty((nr_of_steps))
[x_len,y_len] = data.shape
data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#data = SVF.average_svf(data,gr_SVF)
#[data,data_water] = SVF.MediateData(data,data_water,gr_SVF*gridboxsize,gr_SVF*gridboxsize,gr_SVF*gridboxsize,gridboxsize)
[Roof_frac,Wall_frac,Road_frac,Water_frac,Ground_frac] = SVF.geometricProperties(data,data_water,gridratio,gridboxsize*gr_SVF)
[x_len,y_len] = data.shape
SF_roof=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
SF_wall=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
SF_road=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
#print(SF_roof.shape)
hour_list = []
mean_SFGR5 = []
mean_SF_05 = []
mean_SFGR25 = []
for t in range(nr_of_steps):
    hour = t*(Constants.timestep/3600)%24
    day = (t*(Constants.timestep/3600)//24)+200
    Azi[t],El[t] = Sunpos.solarpos(day,Constants.latitude,Constants.long_rd,hour,radians=True)
    Zenith[t] = np.pi/2-El[t]
    if np.logical_and(hour>=6,hour<=20):
        with open('Pickles/1MaySF/SF_may1_'+str(int(hour))+'_HN1.pickle', 'rb') as f:
            SF_matrix = pickle.load(f)
        # with open('SF1May_aveNM_GR5/SFP1_GR5_NM_' + str(int(hour)) + '.npy', 'rb') as f:
        #     SF_matrix = np.load(f)
        # with open('SF1May_aveNM_GR25/SFP1_GR25_NM_' + str(int(hour)) + '.npy', 'rb') as f:
        #     SF_matrixGR25 = np.load(f)
        # mean_SFGR5.append(np.mean(SF_matrixGR5))
        # mean_SF_05.append(np.mean(SF_matrix))
        # mean_SFGR25.append(np.mean(SF_matrixGR25))
        # hour_list.append(hour)
        #print(SF_matrix.shape)
        [SF_roof[t,:,:],SF_road[t,:,:]] = SVF.average_surfacetype(SF_matrix,data,int(gridratio))
        SF_wall[t,:,:] = SVF.WallSF_fit(Zenith[t],SF_road[t,:,:])
# plt.figure()
# plt.plot(hour_list,mean_SFGR5,label='Mean SF for 2.5m LES averaged grid')
# plt.plot(hour_list,mean_SFGR25,label='Mean SF for 12.5m LES averaged grid')
# plt.plot(hour_list,mean_SF_05,label='Mean SF for 0.5m grid')
# plt.xlabel('time [h]')
# plt.ylabel('mean SF [0-1]')
# plt.legend()
# plt.show()
# print(hour_list)
# print(mean_SF)

"Add noise to signal to make it more realistic"
noise_T = np.random.normal(0, 0.5, nr_of_steps)
noise_SW = np.random.normal(0, 20, nr_of_steps)
noise_T[:2:] = 0
noise_SW[:2:] = 0
# noise_T = 0
# noise_SW = 0

"A first degree fit of all short wave radiation versus zenith angles are computed," \
"this results in the following SW vs Zenith angle distribution:"
a = 1005.792
b = -644.159
SW_down = a + b*Zenith + noise_SW
SW_down[SW_down<0] = 0
SW_dir = SW_down*np.cos(Zenith)
SW_dif = SW_down*(1-np.cos(Zenith))

"LW radiation is based on Air temperature forcing"
sigma = Constants.sigma
time = np.linspace(0,nr_of_steps,nr_of_steps)
q_first_layer = np.ones(len(time)) + 5
T_2m = (np.sin(-np.pi + 2*np.pi/(24*(3600/Constants.timestep))*time)*Constants.T_air_amp)+Constants.T_air + noise_T
WVP = Functions.q_sat(T_2m,Constants.p_atm)*Constants.RH
eps = 1-(1+Constants.c*WVP/T_2m)*np.exp(-np.sqrt(1.2+3*Constants.c*WVP/T_2m))
LW_down = sigma*eps*T_2m**4
# plt.figure()
# plt.plot(time/6,LW_down,label='LW')
# plt.plot(time/6,SW_dif,label='diffuse SW')
# plt.plot(time/6,SW_dir,label='direct SW')
# plt.plot(time/6,SW_down,label='Total SW')
# plt.ylabel('Flux [W/m2]')
# plt.xlabel('time [h]')
# plt.legend()
# plt.title('Forcing LW and SW fluxes')
# plt.show()

"SVF"
#with open('SVF_MatrixP1_GR5_newMethod.npy', 'rb') as f:
with open('SVF_05Matrix.npy', 'rb') as f:
    SVF_matrix = np.load(f)
[SVF_roof,SVF_road] = SVF.average_surfacetype(SVF_matrix,data,gridratio)
SVF_wall = SVF.Inv_WallvsRoadMasson(SVF_road)

"Now all functions"
[T_roof, T_wall,T_road,T_water,T_ground,T_ave_surf,\
           LW_net_roof, SW_net_roof, G_out_surf_roof, SHF_roof, LHF_roof, \
           LW_net_wall, SW_net_wall, G_out_surf_wall, \
           LW_net_road, SW_net_road, G_out_surf_road, SHF_road, LHF_road, \
           LW_net_water, SW_net_water, G_out_surf_water, SHF_water, LHF_water, \
           SW_ave_wall_dif,SW_ave_wall_dir,SW_ave_wall_wall,SW_ave_wall_roof,SW_ave_wall_road] = \
    Functions.HeatEvolution(nr_of_steps,Constants.timestep,
                            SW_down,Zenith,LW_down,T_2m,q_first_layer,
                            SVF_roof,SVF_wall,SVF_road,SF_roof,SF_wall,SF_road,
                            Roof_frac,Road_frac,Wall_frac,Water_frac,
                            Constants.res_roof, Constants.res_road)
#Functions.PlotSurfaceTemp(T_roof,T_wall,T_road,T_water,T_ground,T_2m,T_ave_surf,nr_of_steps)
# Functions.PlotTempLayers(T_wall,T_2m,nr_of_steps)
# Functions.PlotTempLayers(T_ground,T_2m,nr_of_steps)
# Functions.PlotTempLayers(T_roof,T_2m,nr_of_steps)
#Functions.PlotSurfaceFluxes(nr_of_steps,SHF_roof,"SHF_roof",SHF_road,"SHF_road",LHF_roof,"LHF_roof",LHF_road,"LHF_road")
#Functions.PlotSurfaceFluxes(nr_of_steps,LW_net_roof,"Roof",LW_net_wall,"Wall",LW_net_road,"Road",LW_net_water,"Water")
#Functions.PlotSurfaceFluxes(nr_of_steps,SW_net_roof,"Roof",SW_net_wall,"Wall",SW_net_road,"Road",SW_net_water,"Water")
#Functions.PlotSurfaceFluxes(nr_of_steps,SW_ave_wall_dif,"Absorbed Diffuse",SW_ave_wall_dir,"Absorbed Direct",SW_ave_wall_roof,"From Roof",SW_ave_wall_wall,"From Wall",SW_ave_wall_road,"From Road",SW_net_wall,"Net SW", SW_down,"SW received for Wall")

plt.show()
#np.save('AeroRes/AeroRes30/', T_wall)

# layers = 20
# time = (np.arange(nr_of_steps)* Constants.timestep/3600)
# T_asp = Functions.NumericalSoil(time,Constants.timestep,0.03,Constants.lamb_asphalt,Constants.C_asphalt,layers,T_2m)
# T_grass = Functions.NumericalSoil(time,Constants.timestep,0.03,Constants.lamb_grass,Constants.C_grass,layers,T_2m)
# lines = ['solid',(1,(1,1)),(1,(1,2)),(1,(1,3)),(1,(1,4)),(1,(1,5)),(1,(1,6)),(1,(1,7))]
# plt.figure()
# plt.plot(time,T_asp[:,0],'k',linestyle=lines[0])
# plt.plot(time,T_grass[:,0],'g',linestyle=lines[0])
# for l in range(layers):
#     if l%4==0:
#         plt.plot(time,T_asp[:,l],'k',linestyle=lines[int(l/4)])
#         plt.plot(time,T_grass[:,l],'g',linestyle=lines[int(l/4)])
# plt.xlabel('time [h]')
# plt.ylabel('Temperature [K]')
# #plt.legend()
# plt.show()
# with open('AeroResTemps/LHF_roof_30.npy', 'rb') as f:
#     LHF_roof_30 = np.load(f)
# with open('AeroResTemps/LHF_road_30.npy', 'rb') as f:
#     LHF_road_30 = np.load(f)
# with open('AeroResTemps/LHF_roof_60.npy', 'rb') as f:
#     LHF_roof_60 = np.load(f)
# with open('AeroResTemps/LHF_road_60.npy', 'rb') as f:
#     LHF_road_60 = np.load(f)
# with open('AeroResTemps/LHF_roof_90.npy', 'rb') as f:
#     LHF_roof_90 = np.load(f)
# with open('AeroResTemps/LHF_road_90.npy', 'rb') as f:
#     LHF_road_90 = np.load(f)
# time = (np.arange(nr_of_steps)* Constants.timestep/3600)
# # #
# plt.figure()
# plt.plot(time,LHF_road_90,'b', label="Road, 90 s/m")
# plt.plot(time,LHF_road_60,'r', label="Road, 60 s/m")
# plt.plot(time,LHF_road_30,'y', label="Road, 30 s/m")
#plt.plot(time,SHF_road_90,'b--', label="Road, 90 s/m")
#plt.plot(time,SHF_roof_60,'r', label="Roof, 60 s/m")
#plt.plot(time,SHF_road_60,'r--', label="Road, 60 s/m")
#plt.plot(time,SHF_roof_30,'y', label="Roof, 30 s/m")
#plt.plot(time,SHF_road_30,'y--', label="Road, 30 s/m")
# plt.plot(time,T_roof_90[:,0],'y', label="90 s/m")
# plt.plot(time,T_roof_05[:,0],'r--', label="Roof")
# plt.plot(time,T_wall_05[:,0],'b--', label="Wall")
# plt.plot(time,T_road_05[:,0],'y--', label="Road")
# plt.plot(time,T_2m, 'blue', label="Temp at 2m (Forcing)")
# plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
# plt.xlabel("Time [h]")
# #plt.ylabel("Ground Surface Temperature [K]")
# plt.ylabel("LHF [W/m2K]")
# plt.ylim((-50,50))
# plt.legend(loc='upper right')
# plt.show()
