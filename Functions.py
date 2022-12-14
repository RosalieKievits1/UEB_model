# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import SVF
from tqdm import tqdm
from numpy import random

"""Read in data"""
data = pd.read_csv("cabauw_2018.csv", sep = ';')
data.head()

"""upward sensible heat flux"""
SHF = data.iloc[: , 32]
"""Relative humidity at 10 m"""
q_first_layer = data.iloc[: , 30]
"""Upward Latent Heat flux"""
LHF = data.iloc[: , 33]
"""upward longwave heat flux"""
LW_up = data.iloc[: , 34]
"""downward longwave heat flux"""
LW_down = data.iloc[: , 35]
"""upward shortwave heat flux"""
SW_up = data.iloc[: , 36]
"""downward shortwave heat flux"""
SW_down = data.iloc[: , 37]
"""solar zenith angle"""
Zenith = data.iloc[: ,38]
"""the temperature at 2 m high (use as ic for surface temp)"""
T_2m = data.iloc[: ,24]
T_air = T_2m[0]
"""Surface pressure"""
p_surf = data.iloc[: ,5]


"Sinusoidal input"
# time = np.linspace(0,Constants.nr_of_steps,Constants.nr_of_steps)
# SW_down = (np.sin(2*np.pi/(24*60)*time)*100)+100 #np.zeros(len(time)) #
# LW_down = (np.sin(2*np.pi/(24*60)*time)*50)+300 # np.zeros(len(time)) #
# T_2m = (np.sin(2*np.pi/(24*60)*time)*10)+273.15+20 #np.ones(len(time)) + 275
# q_first_layer = np.ones(len(time)) + 5

def exner(pressure):
    p_zero = 10e5
    return (pressure/p_zero)**(Constants.R_d/Constants.C_pd)

def q_sat(T,p):
    "e_sat is the saturation vapor pressure, determined by the Clausius-Clapeyron equation"
    "T must be in Kelvin and this returns the e_sat in Pa, and q in kg/kg"
    e_sat = Constants.e_s_T0 * np.log(Constants.L_v/Constants.R_w*(1/Constants.T_0-1/T))
    q_sat = (Constants.eps*e_sat)/(p - (1-Constants.eps)*e_sat)
    return q_sat

def T_pot(T,p):
    p_zero = 10e5
    T_pot = T * (p_zero/p)**0.286
    return T_pot

"""Equations for map model"""
def initialize_map(layers,data):
    """
    :param layers: number of layers for that surface
    :param nr_steps: nr of time steps we want to simulate
    :param T_surf: initial surface temperature
    :param T_inner_bc: initial temperature for most inner boundery layers
    :return: map_t: array of temperatures and layers
    :return: d: empty array of thicknesses of each layer
    :return: lambdas: empty array of lambdas for each layer
    :return: capacities: empty array for capacities of each layer
    """
    [x_len,y_len] = data.shape
    capacities_roof = np.ones([x_len,y_len,layers]) * Constants.C_brick
    capacities_wall = np.ones([x_len,y_len,layers]) * Constants.C_brick
    capacities_road = np.ones([x_len,y_len,layers]) * Constants.C_soil
    lambdas_roof = np.ones([x_len,y_len,layers]) * Constants.lamb_brick
    lambdas_wall = np.ones([x_len,y_len,layers]) * Constants.lamb_brick
    lambdas_road = np.ones([x_len,y_len,layers]) * Constants.lamb_soil
    T_inner_bc_roof = np.ones([x_len,y_len]) * Constants.T_building
    T_inner_bc_wall = np.ones([x_len,y_len]) * Constants.T_building
    T_inner_bc_road = np.ones([x_len,y_len]) * Constants.T_ground
    map_T_roof = np.ndarray([x_len,y_len,layers])
    map_T_wall = np.ndarray([x_len,y_len,layers])
    map_T_road = np.ndarray([x_len,y_len,layers])

    """Roofs"""
    lin_temp_roof = np.linspace(T_air,T_inner_bc_roof,layers)
    lin_temp_roof = np.swapaxes(lin_temp_roof, 0, 2)
    lin_temp_roof = np.swapaxes(lin_temp_roof, 0, 1)
    map_T_roof[:,:,0:layers] = T_air #lin_temp_roof
    capacities_roof[:,:,0] = Constants.C_bitumen
    lambdas_roof[:,:,0] = Constants.lamb_bitumen

    """Roads"""
    lin_temp_wall = np.linspace(T_air,T_inner_bc_wall,layers)
    lin_temp_wall = np.swapaxes(lin_temp_wall, 0, 2)
    lin_temp_wall = np.swapaxes(lin_temp_wall, 0, 1)
    map_T_wall[:,:,0:layers] = T_air #lin_temp_wall
    """Roads"""
    lin_temp_road = np.linspace(T_air,T_inner_bc_road,layers)
    lin_temp_road = np.swapaxes(lin_temp_road, 0, 2)
    lin_temp_road = np.swapaxes(lin_temp_road, 0, 1)
    map_T_road[:,:,0:layers] = T_air #lin_temp_road
    capacities_road[:,:,0] = Constants.C_asphalt
    lambdas_road[:,:,0] = Constants.lamb_asphalt

    """Albedos and emissivities"""
    emissivity_roof = np.ones([x_len,y_len]) * Constants.e_bitumen
    emissivity_wall = np.ones([x_len,y_len]) * Constants.e_brick
    emissivity_road = np.ones([x_len,y_len]) * Constants.e_asphalt

    albedos_roof = np.ones([x_len,y_len]) * Constants.a_bitumen
    albedos_wall = np.ones([x_len,y_len]) * Constants.a_brick
    albedos_road = np.ones([x_len,y_len]) * Constants.a_asphalt

    return map_T_roof,map_T_wall,map_T_road, \
           capacities_roof,capacities_wall,capacities_road, \
           emissivity_roof,emissivity_wall,emissivity_road, \
           albedos_roof,albedos_wall,albedos_road, \
           lambdas_roof, lambdas_wall, lambdas_road, \
           T_inner_bc_roof, T_inner_bc_wall, T_inner_bc_road

def surfacebalance(albedos_roof, albedos_wall, albedos_road,
                   emissivities_roof, emissivities_wall, emissivities_road,
                   capacities_roof, capacities_wall, capacities_road,
                   SVF_roof, SVF_wall, SVF_road,
                   SF_roof, SF_wall, SF_road,
                   d_roof, d_wall, d_road,
                   lambdas_roof, lambdas_wall, lambdas_road,
                   T_old_roof, T_old_wall, T_old_road,
                   T_old_subs_roof, T_old_subs_wall, T_old_subs_road,
                   delta_t,
                   theta_z,
                   sigma,
                   SW_diff, SW_dir,     # from dales
                   T_firstlayer, q_firstlayer, # from dales
                   LW_down):
    """
    Returns a map of the surface temperatures for all three surface types
    """
    WVF_roof = 1-SVF_roof
    GVF_wall = SVF_wall
    WVF_wall = 1-2*SVF_wall
    WVF_road = 1-SVF_road

    """Longwave radiation"""
    LW_net_roof = LW_down * emissivities_roof * SVF_roof - emissivities_roof * T_old_roof**4 * sigma + \
                  (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_roof * emissivities_roof
    LW_net_wall = LW_down * emissivities_wall * SVF_wall - emissivities_wall * T_old_wall**4 * sigma + \
                  (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_wall * emissivities_wall + \
                  (LW_down * SVF_road * (1-emissivities_road) + emissivities_road * T_old_road**4 * sigma) * GVF_wall * emissivities_wall
    LW_net_road = LW_down * emissivities_road * SVF_road - emissivities_road * T_old_road**4 * sigma + \
                  (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_road * emissivities_road

    " Short wave radiation"
    SW_net_roof = SW_dir * SF_roof * (1-albedos_roof) + SW_diff * SVF_roof * (1-albedos_roof) + \
                  (SW_dir * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_roof) * WVF_roof
    SW_net_wall = SW_dir * SF_wall * (1-albedos_wall) + SW_diff * SVF_wall * (1-albedos_wall) + \
                  (SW_dir * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_wall) * WVF_wall + \
                  (SW_dir * SF_road + SW_diff * SVF_road) * albedos_road * (1-albedos_wall) * GVF_wall
    SW_net_road = SW_dir * SF_road * (1-albedos_road) + SW_diff * SVF_road * (1-albedos_road) + \
                  (SW_dir * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_road) * (1-SVF_road)

    " Latent and Sensible Heat fluxes "
    SHF_roof = Constants.C_pd * Constants.rho_air * (1/Constants.res_roof) * (T_pot(T_old_roof,Constants.p_atm) - T_pot(T_firstlayer,Constants.p_atm))
    SHF_wall = 0
    SHF_road = Constants.C_pd * Constants.rho_air * (1/Constants.res_road) * (T_pot(T_old_road,Constants.p_atm) - T_pot(T_firstlayer,Constants.p_atm))

    LHF_roof = Constants.L_v * Constants.rho_air * (1/Constants.res_roof) * (q_sat(T_old_roof,Constants.p_atm) - q_sat(T_firstlayer,Constants.p_atm))
    LHF_wall = 0
    LHF_road = Constants.L_v * Constants.rho_air * (1/Constants.res_road) * (q_sat(T_old_road,Constants.p_atm) - q_sat(T_firstlayer,Constants.p_atm))

    # print("SHF = " + str(SHF_roof))
    # print("LHF = " + str(LHF_roof))
    " conduction"
    lamb_ave_out_surf_roof = (d_roof[0]+d_roof[1])/((d_roof[0]/lambdas_roof[:,:,0])+(d_roof[1]/lambdas_roof[:,:,1]))
    G_out_surf_roof = lamb_ave_out_surf_roof*((T_old_roof-T_old_subs_roof)/(1/2*(d_roof[0]+d_road[1])))
    lamb_ave_out_surf_wall = (d_wall[0]+d_wall[1])/((d_wall[0]/lambdas_wall[:,:,0])+(d_wall[1]/lambdas_wall[:,:,1]))
    G_out_surf_wall = lamb_ave_out_surf_wall*((T_old_wall-T_old_subs_wall)/(1/2*(d_wall[0]+d_wall[1])))
    lamb_ave_out_surf_road = (d_road[0]+d_road[1])/((d_road[0]/lambdas_road[:,:,0])+(d_wall[1]/lambdas_road[:,:,1]))
    G_out_surf_road = lamb_ave_out_surf_road*((T_old_road-T_old_subs_road)/(1/2*(d_road[0]+d_road[1])))

    " Net radiation "
    netRad_roof = LW_net_roof + SW_net_roof - G_out_surf_roof - SHF_roof - LHF_roof
    netRad_wall = LW_net_wall + SW_net_wall - G_out_surf_wall - SHF_wall - LHF_wall
    netRad_road = LW_net_road + SW_net_road - G_out_surf_road - SHF_road - LHF_road

    " Temperature change "
    dT_roof = (netRad_roof/(capacities_roof[:,:,0]*d_roof[0]))*delta_t
    map_T_roof = T_old_roof + dT_roof
    dT_wall = (netRad_wall/(capacities_wall[:,:,0]*d_wall[0]))*delta_t
    map_T_wall = T_old_wall + dT_wall
    dT_road = (netRad_road/(capacities_road[:,:,0]*d_road[0]))*delta_t
    map_T_road = T_old_road + dT_road

    return map_T_roof,map_T_wall,map_T_road, LW_net_roof, SW_net_roof, LHF_roof, SHF_roof, G_out_surf_roof

def layer_balance(d_roof, d_wall, d_road,
                  lambdas_roof, lambdas_wall, lambdas_road,
                  map_T_roof, map_T_wall, map_T_road,
                  T_old_roof, T_old_wall, T_old_road,
                  T_inner_bc_roof, T_inner_bc_wall,
                  capacities_roof, capacities_wall, capacities_road,
                  layers, delta_t):

    "Returns the temperature change in all layers beneath the surface layer"
    for l in range(1,layers):
        lamb_ave_in_roof = (d_roof[l]+d_roof[l-1])/((d_roof[l-1]/lambdas_roof[:,:,l-1])+(d_roof[l]/lambdas_roof[:,:,l]))
        G_in_roof = lamb_ave_in_roof*((T_old_roof[:,:,l-1]-T_old_roof[:,:,l])/(1/2*(d_roof[l-1]+d_roof[l])))
        lamb_ave_in_wall = (d_wall[l]+d_wall[l-1])/((d_wall[l-1]/lambdas_wall[:,:,l-1])+(d_wall[l]/lambdas_wall[:,:,l]))
        G_in_wall = lamb_ave_in_wall*((T_old_wall[:,:,l-1]-T_old_wall[:,:,l])/(1/2*(d_wall[l-1]+d_wall[l])))
        lamb_ave_in_road = (d_road[l]+d_road[l-1])/((d_road[l-1]/lambdas_road[:,:,l-1])+(d_road[l]/lambdas_road[:,:,l]))
        G_in_road = lamb_ave_in_road*((T_old_road[:,:,l-1]-T_old_road[:,:,l])/(1/2*(d_road[l-1]+d_road[l])))

        """for all layers before the last layer and after the first (surface) layer"""
        if (l < Constants.layers-1):
            lamb_ave_out_roof = (d_roof[l]+d_roof[l+1])/((d_roof[l]/lambdas_roof[:,:,l])+(d_roof[l+1]/lambdas_roof[:,:,l+1]))
            """compute the convective fluxes in and out the layer"""
            G_out_roof = lamb_ave_out_roof*((T_old_roof[:,:,l]-T_old_roof[:,:,l+1])/(1/2*(d_roof[l]+d_roof[l+1])))
            lamb_ave_out_wall = (d_wall[l]+d_wall[l+1])/((d_wall[l]/lambdas_wall[:,:,l])+(d_wall[l+1]/lambdas_wall[:,:,l+1]))
            """compute the convective fluxes in and out the layer"""
            G_out_wall = lamb_ave_out_wall*((T_old_wall[:,:,l]-T_old_wall[:,:,l+1])/(1/2*(d_wall[l]+d_wall[l+1])))
            lamb_ave_out_road = (d_road[l]+d_road[l+1])/((d_road[l]/lambdas_road[:,:,l])+(d_road[l+1]/lambdas_road[:,:,l+1]))
            """compute the convective fluxes in and out the layer"""
            G_out_road = lamb_ave_out_road*((T_old_road[:,:,l]-T_old_road[:,:,l+1])/(1/2*(d_road[l]+d_road[l+1])))
        if (l == layers-1):
            G_out_roof = lambdas_roof[:,:,l]*(T_old_roof[:,:,l]-T_inner_bc_roof)/(1/2*d_roof[l])
            G_out_wall = lambdas_wall[:,:,l]*(T_old_wall[:,:,l]-T_inner_bc_wall)/(1/2*d_wall[l])
            G_out_road = 0
        """Change in temperature"""
        dT_roof = ((G_in_roof-G_out_roof)*delta_t)/(capacities_roof[:,:,l]*d_roof[l])
        dT_wall = ((G_in_wall-G_out_wall)*delta_t)/(capacities_wall[:,:,l]*d_wall[l])
        dT_road = ((G_in_road-G_out_road)*delta_t)/(capacities_road[:,:,l]*d_road[l])

        "New temperatures"
        map_T_roof[:,:,l] = T_old_roof[:,:,l] + dT_roof
        map_T_wall[:,:,l] = T_old_wall[:,:,l] + dT_wall
        map_T_road[:,:,l] = T_old_road[:,:,l] + dT_road

    return map_T_roof[:,:,1:layers], map_T_wall[:,:,1:layers], map_T_road[:,:,1:layers]

def HeatEvolution(data,time_steps,delta_t):

    "Arrays that store average surface temperatures"
    T_ave_roof = np.empty((time_steps,Constants.layers))
    T_ave_wall = np.empty((time_steps,Constants.layers))
    T_ave_road = np.empty((time_steps,Constants.layers))
    LW_ave = np.empty((time_steps))
    SW_ave = np.empty((time_steps))
    G_ave = np.empty((time_steps))
    SHF_ave = np.empty((time_steps))
    LHF_ave = np.empty((time_steps))
    "Compute the surface fractions"
    # [wallArea_matrix, wallArea_total] = SVF.wallArea(data)
    # wall_layers = np.ndarray([wallArea_matrix.shape[0],wallArea_matrix.shape[1],Constants.layers])

    "Compute the shadowfactor and SVF"
    #coords = SVF.coordheight(data)
    #[svf,Shadowfactor] = SVF.reshape_SVF(data, coords,azimuth,elevation_angle,reshape=True,save_CSV=False,save_Im=False)
    #[SVF_roof, SVF_road] = SVF.average_svf_surfacetype(svf,data,10)
    "Now we need to separate the roof, wall and road SVF and SF"
    [x_len,y_len] = data.shape
    "All roofs receive sunlight, 30% of the ground and 40% of the walls receive sunlight"

    """We evaluate the middle block only, but after the SVF and Shadowfactor are calculated"""
    #[x_len,y_len] = data.shape
    #data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]

    map_T_roof,map_T_wall,map_T_road, \
           capacities_roof,capacities_wall,capacities_road, \
           emissivity_roof,emissivity_wall,emissivity_road, \
           albedos_roof,albedos_wall,albedos_road, \
           lambdas_roof, lambdas_wall, lambdas_road, \
           T_inner_bc_roof, T_inner_bc_wall, T_inner_bc_road = initialize_map(Constants.layers,data)

    map_T_old_roof = map_T_roof
    map_T_old_wall = map_T_wall
    map_T_old_road = map_T_road

    "Height over width ratio"
    h_w = 0.7
    SVF_Roof = 1
    SVF_Wall = 1/2 *(h_w+1-np.sqrt(h_w**2+1))/h_w
    SVF_Road = np.sqrt(h_w**2+1)-h_w
    "zenith angle"

    for t in tqdm(range(time_steps)):
        zenith = Zenith[t]/180*np.pi
        theta_zero = np.arcsin(min(1/h_w*1/np.tan(zenith),1))
        SF_Roof = 1
        SF_Wall = 1/h_w*(1/2-theta_zero/np.pi)+1/np.pi*np.tan(zenith)*(1-np.cos(theta_zero))
        SF_Road = 2*theta_zero/np.pi - 2/np.pi*h_w*np.tan(zenith)*(1-np.cos(theta_zero))
        SF_roof = np.ones([x_len,y_len])*SF_Roof
        SF_wall = np.ones([x_len,y_len])*SF_Wall
        SF_road = np.ones([x_len,y_len])*SF_Road
        SVF_Roof = np.ones([x_len,y_len])*SVF_Roof
        SVF_Wall = np.ones([x_len,y_len])*SVF_Wall
        SVF_Road = np.ones([x_len,y_len])*SVF_Road
        SW_dir = SW_down[t]*0.3
        SW_dif = SW_down[t]*0.7
        LW_d = LW_down[t]
        "Set these to constants for now"
        T_firstlayer = T_2m[t]
        q_firstlayer = q_first_layer[t]
        [map_T_roof[:,:,0],map_T_wall[:,:,0],map_T_road[:,:,0], LW_net_roof, SW_net_roof, LHF_roof, SHF_roof, G_out_surf_roof] = surfacebalance(albedos_roof, albedos_wall, albedos_road, \
            emissivity_roof, emissivity_wall, emissivity_road, \
            capacities_roof, capacities_wall, capacities_road, \
            SVF_Roof, SVF_Wall, SVF_Road, \
            SF_roof, SF_wall, SF_road, \
            Constants.d_roof, Constants.d_wall, Constants.d_road, \
            lambdas_roof, lambdas_wall, lambdas_road, \
            map_T_old_roof[:,:,0], map_T_old_wall[:,:,0], map_T_old_road[:,:,0], \
            map_T_roof[:,:,1], map_T_wall[:,:,1], map_T_road[:,:,1], \
            delta_t, \
            zenith, \
            Constants.sigma, \
            SW_dif, SW_dir, \
            T_firstlayer, q_firstlayer,\
            LW_d)

        map_T_roof[:,:,1:], map_T_wall[:,:,1:], map_T_road[:,:,1:] = layer_balance(Constants.d_roof, Constants.d_wall, Constants.d_road,
                  lambdas_roof, lambdas_wall, lambdas_road,
                  map_T_roof, map_T_wall, map_T_road,
                  map_T_old_roof, map_T_old_wall, map_T_old_road,
                  T_inner_bc_roof, T_inner_bc_wall,
                  capacities_roof, capacities_wall, capacities_road,
                  Constants.layers, delta_t)

        map_T_old_roof = map_T_roof
        map_T_old_wall = map_T_wall
        map_T_old_road = map_T_road

        "Average SURFACE temperatures"
        for l in range(Constants.layers):
            T_ave_roof[t,l] = np.mean(map_T_roof[:,:,l])
            T_ave_wall[t,l] = np.mean(map_T_wall[:,:,l])
            T_ave_road[t,l] = np.mean(map_T_road[:,:,l])
        #T_ave_surf[t] = np.mean(T_surf_roof*Roof_frac + T_surf_wall*Wall_frac + T_surf_road*Road_frac)
        LW_ave[t] = np.mean(LW_net_roof)
        SW_ave[t] = np.mean(SW_net_roof)
        LHF_ave[t] = np.mean(LHF_roof)
        SHF_ave[t] = np.mean(SHF_roof)
        G_ave[t] = np.mean(G_out_surf_roof)

    return T_ave_roof, T_ave_wall, T_ave_road, LW_ave, SW_ave, LHF_ave, SHF_ave, G_ave

def AnalyticalSoil(t,z,lamb,C):
    k = lamb/C
    period = 24
    omega = 2*np.pi/(period*3600)
    d = (2*k/omega)**(1/2) # Damping depth
    phi_z = -z/d
    T = 273.15+20+10*np.exp(phi_z)*np.sin(omega*t*Constants.timestep+phi_z)
    return T

def NumericalSoil(time,delta_t,delta_d,lambd,Cap,layers,T_in):
    C = np.ones((layers))*Cap
    lamb = np.ones((layers))*lambd
    T = np.ones([len(time),layers])
    T_old = np.ones((layers))*T_in[0]
    T[0,:] = T_old
    for t in range(len(time)):
        for l in range(layers):
            if l>0:
                lamb_ave_in = 2*delta_d/((delta_d/lamb[l-1])+(delta_d/lamb[l]))
                G_in = lamb_ave_in*(T_old[l-1]-T_old[l])/delta_d
            elif l == 0:
                lamb_ave_in = lamb[0]
                G_in = lamb_ave_in*(T_in[t]-T_old[l])/(1/2*delta_d)
            if l<layers-1:
                lamb_ave_out = 2*delta_d/((delta_d/lamb[l])+(delta_d/lamb[l+1]))
                G_out = lamb_ave_out*(T_old[l]-T_old[l+1])/delta_d
            elif l==layers-1:
                G_out=0
            dT = ((G_in-G_out)*delta_t)/(C[l]*delta_d)
            T[t,l] = T_old[l] + dT
        T_old = T[t,:]
    return T

"""PLOTFUNCTIONS"""
def PlotGreyMap(data,middle,v_max):
    plt.figure()
    plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
    if middle == True:
        [x_len,y_len] = data.shape
        plt.imshow(data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)],vmax=v_max)
    elif middle == False:
        plt.imshow(data,vmax=v_max)
    plt.show()

def PlotSurfaceTemp(T_ave_roof,T_ave_wall,T_ave_road, time_steps):
    time = (np.arange(time_steps))#* Constants.timestep/3600)

    plt.figure()
    plt.plot(time,T_ave_roof[:,0], label="Simulated Surface Temp")
    # plt.plot(time,T_ave_wall[:,0], label="Wall")
    #plt.plot(time,T_ave_road[:,0], label="road")
    plt.plot(time,T_2m[:Constants.nr_of_steps], 'blue', label="Temp at 2m altitude")
    plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
    plt.xlabel("Time [h]")
    plt.ylabel("Surface Temperature [K]")
    plt.legend(loc='upper right')
    plt.show()
    return

def PlotTempLayers(T_ave,time_steps):
    time = (np.arange(time_steps)* Constants.timestep/3600)
    plt.figure()
    for l in range(Constants.layers):
        plt.plot(time,T_ave[:,l], label="layer " + str(l))
    plt.plot(time,T_2m[:Constants.nr_of_steps], label="Measured Temp above surface")
    plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [K]")
    plt.legend(loc='upper right')
    plt.show()
    return

def PlotSurfaceFluxes(LW_net,SW_net,G_out,LHF,SHF):
    time = (np.arange(Constants.nr_of_steps)*Constants.timestep / 3600)
    plt.figure()
    plt.plot(time,LW_net, label="LW net")
    #plt.plot(time,(LW_down[:Constants.nr_of_steps]-LW_up[:Constants.nr_of_steps]), label="LW up cabau")
    plt.plot(time,SW_net, label="SW net")
    #plt.plot(time,(SW_down[:Constants.nr_of_steps]-SW_up[:Constants.nr_of_steps]), label="SW up cabau")
    plt.plot(time,SW_down[:Constants.nr_of_steps],label="SW down")
    # LHF_cabau = Functions.LHF[:Constants.nr_of_steps]
    # LHF_cabau[LHF_cabau<-1000] = 0
    # SHF_cabau = Functions.SHF[:Constants.nr_of_steps]
    # SHF_cabau[SHF_cabau<-1000] = 0
    # plt.plot(time,LHF_cabau,label="LHF_cabau")
    # plt.plot(time,SHF_cabau,label="SHF_cabau")
    plt.plot(time,LHF, label="LHF")
    plt.plot(time,SHF, label="SHF")
    plt.plot(time,G_out, label="G")
    plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
    plt.xlabel("Time [h]")
    plt.ylabel("Flux [W/m2K]")
    plt.legend(loc='upper right')
    plt.show()
    return
