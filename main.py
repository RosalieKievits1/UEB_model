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
gridratio = 5
gr_SVF = 5
gridboxsize = SVF.gridboxsize

"Azimuth and zenith angle based on the day of the year"
days = 6
nr_of_steps = int(24*3600/Constants.timestep*days)
startday = Constants.julianday

Azi = np.empty((nr_of_steps))
El = np.empty((nr_of_steps))
Zenith = np.empty((nr_of_steps))
[x_len,y_len] = data.shape
data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#data = SVF.average_svf(data,gr_SVF)
data = SVF.MediateData(data,gr_SVF*gridboxsize,gr_SVF*gridboxsize,gr_SVF*gridboxsize,gridboxsize)
[Roof_frac,Wall_frac,Road_frac,Water_frac] = SVF.geometricProperties(data,data_water,gridratio,gridboxsize*gr_SVF)
#print(data.shape)
[x_len,y_len] = data.shape
SF_roof=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
SF_wall=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
SF_road=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
#print(SF_roof.shape)
for t in range(nr_of_steps):
    hour = t*(Constants.timestep/3600)%24
    day = (t*(Constants.timestep/3600)//24)+200
    Azi[t],El[t] = Sunpos.solarpos(day,Constants.latitude,Constants.long_rd,hour,radians=True)
    Zenith[t] = np.pi/2-El[t]
    if np.logical_and(hour>=6,hour<=20):
        with open('Pickles/1MaySF/SF_may1_'+str(int(hour))+'_HN1.pickle', 'rb') as f:
            SF_matrix = pickle.load(f)
        [SF_roof[t,:,:],SF_road[t,:,:]] = SVF.average_surfacetype(SF_matrix,data,int(gridratio*gr_SVF))
        SF_wall[t,:,:] = SVF.WallSF_fit(Zenith[t],SF_road[t,:,:])


"A first degree fit of all short wave radiation versus zenith angles are computed," \
"this results in the following SW vs Zenith angle distribution:"
a = 1005.792
b = -644.159
SW_down = a + b*Zenith
SW_down[SW_down<0] = 0

"LW radiation is based on Air temperature forcing"
sigma = Constants.sigma
time = np.linspace(0,nr_of_steps,nr_of_steps)
q_first_layer = np.ones(len(time)) + 5
T_2m = (np.sin(-np.pi + 2*np.pi/(24*(3600/Constants.timestep))*time)*Constants.T_air_amp)+Constants.T_air
WVP = Functions.q_sat(T_2m,Constants.p_atm)*Constants.RH
eps = 1-(1+Constants.c*WVP/T_2m)*np.exp(-np.sqrt(1.2+3*Constants.c*WVP/T_2m))
LW_down = sigma*eps*T_2m**4
# plt.figure()
# plt.plot(time/6,LW_down,label='LW')
# plt.plot(time/6,SW_down,label='SW')
# plt.ylabel('Flux [W/m2]')
# plt.xlabel('time [h]')
# plt.legend()
# plt.title('Forcing LW and SW fluxes')
#plt.show()

"SVF"
with open('SVF_MatrixP1_GR5_newMethod.npy', 'rb') as f:
    SVF_matrix = np.load(f)
[SVF_roof,SVF_road] = SVF.average_surfacetype(SVF_matrix,data,gridratio)
SVF_wall = SVF.Inv_WallvsRoadMasson(SVF_road)

"Now all functions"
[T_roof, T_wall,T_road,T_water,T_ground, LW_net_roof, SW_net_roof, LHF_roof, SHF_roof, G_out_surf_roof] = \
    Functions.HeatEvolution(nr_of_steps,Constants.timestep,
                            SW_down,LW_down,T_2m,q_first_layer,
                            SVF_roof,SVF_wall,SVF_road,SF_roof,SF_wall,SF_road,
                            Roof_frac,Road_frac,Wall_frac,Water_frac)
Functions.PlotSurfaceTemp(T_roof,T_wall,T_road,T_water,T_ground,T_2m,nr_of_steps)
Functions.PlotTempLayers(T_wall,T_2m,nr_of_steps)
Functions.PlotSurfaceFluxes(nr_of_steps,LW_net_roof, SW_net_roof,SW_down, G_out_surf_roof, LHF_roof, SHF_roof,show=True)

# np.save('Temp/T_roof_GR5_NM', T_roof)
# np.save('Temp/T_wall_GR5_NM', T_wall)
# np.save('Temp/T_road_GR5_NM', T_road)

# with open('Temp/T_roof_GR5.npy', 'rb') as f:
#     T_roof_5 = np.load(f)
# with open('Temp/T_roof_GR0.npy', 'rb') as f:
#     T_roof_05 = np.load(f)
# with open('Temp/T_wall_GR5.npy', 'rb') as f:
#     T_wall_5 = np.load(f)
# with open('Temp/T_wall_GR0.npy', 'rb') as f:
#     T_wall_05 = np.load(f)
# with open('Temp/T_road_GR5.npy', 'rb') as f:
#     T_road_5 = np.load(f)
# with open('Temp/T_road_GR0.npy', 'rb') as f:
#     T_road_05 = np.load(f)
# time = (np.arange(nr_of_steps)* Constants.timestep/3600)
# #
# plt.figure()
# plt.plot(time,T_roof_5[:,0]-T_roof_05[:,0],'r', label="Roof, difference")
# plt.plot(time,T_wall_5[:,0]-T_wall_05[:,0],'b', label="Wall, difference")
# plt.plot(time,T_road_5[:,0]-T_road_05[:,0],'y', label="Road, difference")
# #plt.plot(time,T_roof_05[:,0],'r--', label="Roof")
# #plt.plot(time,T_wall_05[:,0],'b--', label="Wall")
# #plt.plot(time,T_road_05[:,0],'y--', label="Road")
# #plt.plot(time,T_2m, 'blue', label="Temp at 2m (Forcing)")
# plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
# plt.xlabel("Time [h]")
# plt.ylabel("Surface Temperature [K]")
# plt.legend(loc='upper right')
# plt.show()
