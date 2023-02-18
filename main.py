# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import Functions
import SVF
import Sunpos
#import pickle

"Data import"
data = SVF.data
gridratio = 25
"Azimuth and zenith angle based on the day of the year"
days = 6
nr_of_steps = int(24*3600/Constants.timestep*days)
startday = Constants.julianday

Azi = np.empty((nr_of_steps))
El = np.empty((nr_of_steps))
Zenith = np.empty((nr_of_steps))
for t in range(nr_of_steps):
    hour = t*(Constants.timestep/3600)%24
    day = (t*(Constants.timestep/3600)//24)+200
    Azi[t],El[t] = Sunpos.solarpos(day,Constants.latitude,Constants.long_rd,hour,radians=True)
    Zenith[t] = np.pi/2-El[t]
    if np.logical_and(hour>=6,hour<=20):
        with open('Pickles/1MaySF/SF_may1_'+str(int(hour))+'_HN1.pickle', 'rb') as f:
            SF_matrix = pickle.load(f)
        [SF_roof,SF_road] = SVF.average_surfacetype(SF_matrix,data,gridratio)
        SF_wall = SVF.WallSF_fit(Zenith[t],SF_road)
    else:
        [x_len,y_len] = data.shape
        SF_roof=np.zeros([int(x_len/2/gridratio),int(y_len/2/gridratio)])
        SF_wall=np.zeros([int(x_len/2/gridratio),int(y_len/2/gridratio)])
        SF_road=np.zeros([int(x_len/2/gridratio),int(y_len/2/gridratio)])

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
eps = 0.7
T_2m = (np.sin(-np.pi + 2*np.pi/(24*(3600/Constants.timestep))*time)*Constants.T_air_amp)+Constants.T_air
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
with open('SVF_05Matrix.npy', 'rb') as f:
    SVF_matrix = np.load(f)
[SVF_roof,SVF_road] = SVF.average_surfacetype(SVF_matrix,data,gridratio)
SVF_wall = SVF.Inv_WallvsRoadMasson(SVF_road)

"Now all functions"
[T_roof, T_wall,T_road, LW_net_roof, SW_net_roof, LHF_roof, SHF_roof, G_out_surf_roof] = \
    Functions.HeatEvolution(nr_of_steps,Constants.timestep,
                            SW_down,LW_down,T_2m,q_first_layer,
                            SVF_roof,SVF_wall,SVF_road,SF_roof,SF_wall,SF_road)
Functions.PlotSurfaceTemp(T_roof,T_wall,T_road,T_2m,nr_of_steps)
Functions.PlotTempLayers(T_wall,T_2m,nr_of_steps)
Functions.PlotSurfaceFluxes(nr_of_steps,LW_net_roof, SW_net_roof,SW_down, G_out_surf_roof, LHF_roof, SHF_roof,show=True)
