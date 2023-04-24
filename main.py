# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
#plt.rc('font', size=15) #for three plots togethes
plt.rc('font', size=12) #for 2 plots together
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

# time_h = (np.arange(nr_of_steps)* Constants.timestep/3600)
# plt.figure()
# plt.xlabel('time [h]')
# plt.xlim((0,24))
# plt.plot(time_h,SW_dir,'orange',label='direct SW')
# plt.plot(time_h,SW_dif,'darkcyan',label='diffuse SW')
# plt.plot(time_h,SW_down,'orangered',label='Total SW')
# plt.plot(time_h,LW_down,'deepskyblue',label='LW')
# plt.ylabel('Flux [W/m^2K]')
# plt.legend()
# plt.show()


"SVF"
#with open('SVF_MatrixP1_GR5_newMethod.npy', 'rb') as f:
with open('SVF_05Matrix.npy', 'rb') as f:
    SVF_matrix = np.load(f)
[SVF_roof,SVF_road] = SVF.average_surfacetype(SVF_matrix,data,gridratio)
SVF_wall = SVF.Inv_WallvsRoadMasson(SVF_road)

"Now all functions"
[T_roof, T_wall,T_road,T_water,T_ground,T_ave_surf,\
           LW_net_roof, SW_net_roof, G_out_surf_roof, SHF_roof, \
           LW_net_wall, SW_net_wall, G_out_surf_wall, SHF_wall, \
           LW_net_road, SW_net_road, G_out_surf_road, SHF_road, \
           LW_net_water, SW_net_water, G_out_surf_water, SHF_water, LHF_water, \
           SW_ground_net, LW_ground_net, SHF_ground, LHF_ground, G_out_surf_ground, \
           SW_r_dir,SW_r_dif,SW_r_wall, \
           SW_g_dir,SW_g_dif,SW_g_wall, \
           SW_wall_dir, SW_wall_dif, SW_wall_wall,SW_wall_roof,SW_wall_road,\
           LW_roof_first, LW_roof_em, LW_roof_w, \
           LW_w_first, LW_w_em, LW_w_wall, LW_w_roof, LW_w_road, \
           LW_ground_first, LW_ground_em, LW_ground_w] = \
    Functions.HeatEvolution(nr_of_steps,Constants.timestep,
                            SW_down,Zenith,LW_down,T_2m,q_first_layer,
                            SVF_roof,SVF_wall,SVF_road,SF_roof,SF_wall,SF_road,
                            Roof_frac,Road_frac,Wall_frac,Water_frac,
                            Constants.res_roof, Constants.res_wall, Constants.res_road)

# "The temperatures"
# np.save('BuildingTemps/05m/Winter/T_air', T_2m)
# np.save('BuildingTemps/05m/Winter/T_ground', T_ground)
# np.save('BuildingTemps/05m/Winter/T_wall', T_wall)
# np.save('BuildingTemps/05m/Winter/T_roof', T_roof)
#
# "The sensible/latent heat fluxes"
# np.save('BuildingTemps/05m/Winter/SHF_roof', SHF_roof)
# np.save('BuildingTemps/05m/Winter/SHF_wall', SHF_wall)
# np.save('BuildingTemps/05m/Winter/SHF_ground', SHF_ground)
# np.save('BuildingTemps/05m/Winter/LHF_ground', LHF_ground)
#
# "LW"
# np.save('BuildingTemps/05m/Winter/LW_net_roof', LW_net_roof)
# np.save('BuildingTemps/05m/Winter/LW_net_wall', LW_net_wall)
# np.save('BuildingTemps/05m/Winter/LW_net_ground', LW_ground_net)
# np.save('BuildingTemps/05m/Winter/LW_down', LW_down)
#
# "SW"
# np.save('BuildingTemps/05m/Winter/SW_net_roof', SW_net_roof)
# np.save('BuildingTemps/05m/Winter/SW_net_wall', SW_net_wall)
# np.save('BuildingTemps/05m/Winter/SW_net_ground', SW_ground_net)
# np.save('BuildingTemps/05m/Winter/SW_down', SW_down)
#
# "Conduction"
# np.save('BuildingTemps/05m/Winter/G_net_roof', G_out_surf_roof)
# np.save('BuildingTemps/05m/Winter/G_net_wall', G_out_surf_wall)
# np.save('BuildingTemps/05m/Winter/G_net_ground', G_out_surf_ground)

Functions.PlotSurfaceTemp(T_roof,T_wall,T_road,T_water,T_ground,T_2m,T_ave_surf,nr_of_steps)
Functions.PlotTempLayers(T_wall,T_2m,nr_of_steps)
Functions.PlotTempLayers(T_ground,T_2m,nr_of_steps)
Functions.PlotTempLayers(T_roof,T_2m,nr_of_steps)
Functions.PlotSurfaceFluxes(nr_of_steps,[SHF_roof,SHF_road,SHF_wall],["SHF_roof","SHF_road","SHF_wall"])
Functions.PlotSurfaceFluxes(nr_of_steps,[LW_net_roof,LW_net_wall,LW_net_road,LW_net_water],["Wall","Roof","Road","Water"])
Functions.PlotSurfaceFluxes(nr_of_steps,[SW_net_roof,SW_net_wall,SW_net_road,SW_net_water,],["Roof","Wall","Road","Water"])
Functions.PlotSurfaceFluxes(nr_of_steps,[SW_r_dif,SW_r_dir,SW_r_wall,SW_net_roof,SW_down],["Absorbed Diffuse","Absorbed Direct","From Wall","Net SW","SW received"])
plt.show()


