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
gridratio = 5
gr_SVF = 1
gridboxsize = SVF.gridboxsize

"Azimuth and zenith angle based on the day of the year"
days = 1
nr_of_steps = int(24*3600/Constants.timestep*days)
startday = Constants.julianday

Azi = np.empty((nr_of_steps))
El = np.empty((nr_of_steps))
Zenith = np.empty((nr_of_steps))
[x_len,y_len] = data.shape
data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#data = SVF.average_svf(data,gr_SVF)
#[data,data_water] = SVF.MediateData(data,data_water,gr_SVF*gridboxsize,gr_SVF*gridboxsize,gr_SVF*gridboxsize,gridboxsize)
"GR 25"
[data,data_water] = SVF.MediateData(data,data_water,12.5,12.5,10,gridboxsize)
[Roof_frac,Wall_frac,Road_frac,Water_frac,Ground_frac] = SVF.geometricProperties(data,data_water,gridratio,gridboxsize*gr_SVF)
[x_len,y_len] = data.shape
SF_roof=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
SF_wall=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
SF_road=np.zeros([nr_of_steps, int(x_len/int(gridratio)),int(y_len/int(gridratio))])
for t in range(nr_of_steps):
    hour = t*(Constants.timestep/3600)%24
    day = (t*(Constants.timestep/3600)//24)+200
    Azi[t],El[t] = Sunpos.solarpos(day,Constants.latitude,Constants.long_rd,hour,radians=True)
    Zenith[t] = np.pi/2-El[t]
    if np.logical_and(hour>=6,hour<=20):
        with open('SFmatrices/Pickles/1MaySF/SF_may1_'+str(int(hour))+'_HN1.pickle', 'rb') as f:
            SF_matrix = pickle.load(f)
        # with open('SFmatrices/SF1May_aveNM_GR5/SFP1_GR5_NM_' + str(int(hour)) + '.npy', 'rb') as f:
        #     SF_matrixGR5 = np.load(f)



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


"SVF"
#with open('SVF_MatrixP1_GR5_newMethod.npy', 'rb') as f:
#with open('SVF_05Matrix.npy', 'rb') as f:
with open('SVFmatrices/SVF_MatrixP1_GR25_newMethod.npy', 'rb') as f:
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



