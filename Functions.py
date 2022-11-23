# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import SVF

"""Read in data"""
data = pd.read_csv("cabauw_2018.csv", sep = ';')
data.head()

"""upward sensible heat flux"""
SHF = data.iloc[: , 32]
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
"""Surface pressure"""
p_surf = data.iloc[: ,5]
"""nr of steps"""
nr_steps = 100 #np.size(LW_up,0)

T_air = T_2m[0]

def exner(pressure):
    p_zero = 10e5
    return (pressure/p_zero)**(Constants.R_d/Constants.C_pd)

def q_sat(T,p):
    R_w = 461.52 # J/kgK gas constant of water
    L = 2.5e6 #J/kg latent vaporization heat of water
    T_0 = 273.16 # K ref temp
    e_s_T0 = 6.11e2 #Pa e_s at reference temperature
    eps = 0.622 # ratio of molar masses of vapor and dry air

    e_sat = e_s_T0 * np.log(L/R_w*(1/T_0-1/T))
    q_sat = (eps*e_sat)/(p - (1-eps)*e_sat)
    return q_sat


"""Equations for map model"""
def initialize_map(layers,T_surf,data):
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
    capacities_roof = np.ones([x_len,y_len,layers]) * Constants.C_bitumen
    capacities_wall = np.ones([x_len,y_len,layers]) * Constants.C_brick
    capacities_road = np.ones([x_len,y_len,layers]) * Constants.C_asphalt
    lambdas_roof = np.ones([x_len,y_len,layers]) * Constants.lamb_bitumen
    lambdas_wall = np.ones([x_len,y_len,layers]) * Constants.lamb_brick
    lambdas_road = np.ones([x_len,y_len,layers]) * Constants.lamb_asphalt
    T_inner_bc_roof = np.ones([x_len,y_len]) * Constants.T_building
    T_inner_bc_wall = np.ones([x_len,y_len]) * Constants.T_building
    T_inner_bc_road = np.ones([x_len,y_len]) * Constants.T_ground
    map_T_roof = np.ones([x_len,y_len,layers])
    map_T_wall = np.ones([x_len,y_len,layers])
    map_T_road = np.ones([x_len,y_len,layers])

    """Roofs"""
    lin_temp_roof = np.linspace(T_surf,T_inner_bc_roof,layers)
    map_T_roof[:,:,0:layers-1] = lin_temp_roof
    """Roads"""
    lin_temp_wall = np.linspace(T_surf,T_inner_bc_wall,layers)
    map_T_wall[:,:,0:layers-1] = lin_temp_wall
    """Roads"""
    lin_temp_road = np.linspace(T_surf,T_inner_bc_road,layers)
    map_T_wall[:,:,0:layers-1] = lin_temp_road

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
                   WVF_wall, GVF_wall,
                   SF_roof, SF_wall, SF_road,
                   d_roof, d_wall, d_road,
                   lambdas_roof, lambdas_wall, lambdas_road,
                   T_old_roof, T_old_wall, T_old_road,
                   T_old_subs_roof, T_old_subs_wall, T_old_subs_road,
                   delta_t,
                   sigma,
                   theta_z,
                   SW_diff, SW_dir,
                   LW_down):
    """
    Returns a map of the surface temperatures for all three surface types
    """

    """Longwave radiation"""
    LW_net_roof = LW_down * emissivities_roof * SVF_roof - emissivities_roof * T_old_roof**4 * sigma + \
                  (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * (1-SVF_roof) * emissivities_roof
    LW_net_wall = LW_down * emissivities_wall * SVF_wall - emissivities_wall * T_old_wall * sigma + \
                  (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_wall * emissivities_wall + \
                  (LW_down * SVF_road * (1-emissivities_road) + emissivities_road * T_old_road**4 * sigma) * GVF_wall * emissivities_wall
    LW_net_road = LW_down * emissivities_road * SVF_road - emissivities_road * T_old_road**4 * sigma + \
                  (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * (1-SVF_road) * emissivities_road

    """ Short wave radiation"""
    SW_net_roof = SW_dir * SF_roof * (1-albedos_roof) + SW_diff * SVF_roof * (1-albedos_roof) + \
                  (SW_dir * np.tan(theta_z) * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_roof) * (1-SVF_roof)
    SW_net_wall = SW_dir * np.tan(theta_z) * SF_wall * (1-albedos_wall) + SW_diff * SVF_wall * (1-albedos_wall) + \
                  (SW_dir * np.tan(theta_z) * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_wall) * WVF_wall + \
                  (SW_dir * SF_road + SW_diff * SVF_road) * albedos_road * (1-albedos_wall) * GVF_wall
    SW_net_road = SW_dir * SF_road * (1-albedos_road) + SW_diff * SVF_road * (1-albedos_road) + \
                  (SW_dir * np.tan(theta_z) * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_road) * (1-SVF_road)

    """conduction"""
    lamb_ave_out_surf_roof = (d_roof[0]+d_roof[1])/((d_roof[0]/lambdas_roof[:,:,0])+(d_roof[1]/lambdas_roof[:,:,1]))
    G_out_surf_roof = lamb_ave_out_surf_roof*((T_old_roof-T_old_subs_roof)/(1/2*(d_roof[0]+d_road[1])))
    lamb_ave_out_surf_wall = (d_wall[0]+d_wall[1])/((d_wall[0]/lambdas_wall[:,:,0])+(d_wall[1]/lambdas_wall[:,:,1]))
    G_out_surf_wall = lamb_ave_out_surf_wall*((T_old_wall-T_old_subs_wall)/(1/2*(d_wall[0]+d_wall[1])))
    lamb_ave_out_surf_road = (d_road[0]+d_road[1])/((d_road[0]/lambdas_road[:,:,0])+(d_wall[1]/lambdas_road[:,:,1]))
    G_out_surf_road = lamb_ave_out_surf_road*((T_old_road-T_old_subs_road)/(1/2*(d_road[0]+d_road[1])))

    """ Net radiation"""
    netRad_roof = LW_net_roof + SW_net_roof - G_out_surf_roof
    netRad_wall = LW_net_wall + SW_net_wall - G_out_surf_wall
    netRad_road = LW_net_road + SW_net_road - G_out_surf_road

    """ Temperature change"""
    dT_roof = (netRad_roof/(capacities_roof[:,:,0]*d_roof[0]))*delta_t
    map_T_roof = T_old_roof + dT_roof
    dT_wall = (netRad_wall/(capacities_wall[:,:,0]*d_wall[0]))*delta_t
    map_T_wall = T_old_wall + dT_wall
    dT_road = (netRad_road/(capacities_road[:,:,0]*d_road[0]))*delta_t
    map_T_road = T_old_road + dT_road

    return map_T_roof,map_T_wall,map_T_road

def layer_balance(d_roof, d_wall, d_road,
                  lambdas_roof, lambdas_wall, lambdas_road,
                  map_T_roof, map_T_wall, map_T_road,
                  T_old_roof, T_old_wall, T_old_road,
                  T_inner_bc_roof, T_inner_bc_wall,
                  capacities_roof, capacities_wall, capacities_road,
                  layers, delta_t):
    "Returns the temperature change in all layers beneath the surface layer"

    # probably not necessary to initialize?
    # G_out_roof = np.ndarray(map_T_roof.shape)
    # G_out_wall = np.ndarray(map_T_wall.shape)
    # G_out_road = np.ndarray(map_T_road.shape)

    for l in range(1,layers):
        lamb_ave_in_roof = (d_roof[l]+d_roof[l-1])/((d_roof[l-1]/lambdas_roof[:,:,l-1])+(d_roof[l]/lambdas_roof[:,:,l]))
        G_in_roof = lamb_ave_in_roof*((T_old_roof[:,:,l-1]-T_old_roof[:,:,l])/(1/2*(d_roof[l-1]+d_roof[l])))
        lamb_ave_in_wall = (d_wall[l]+d_wall[l-1])/((d_wall/lambdas_wall[:,:,l-1])+(d_wall[l]/lambdas_wall[:,:,l]))
        G_in_wall = lamb_ave_in_wall*((T_old_wall[:,:,l-1]-T_old_wall[:,:,l])/(1/2*(d_wall[l-1]+d_wall[l])))
        lamb_ave_in_road = (d_road[l]+d_road[l-1])/((d_road/lambdas_road[:,:,l-1])+(d_road[l]/lambdas_road[:,:,l]))
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

def HeatEvolution(data,time_steps,delta_t,azimuth,elevation_angle, T_surf):

    "Arrays that store average surface temperatures"
    T_ave_roof = np.array([time_steps])
    T_ave_wall = np.array([time_steps])
    T_ave_road = np.array([time_steps])

    "Compute the surface fractions"
    [wallArea_matrix, wallArea_total] = SVF.wallArea(data)
    wall_layers = np.ndarray([wallArea_matrix.shape[0],wallArea_matrix.shape[1],Constants.layers])

    "Compute the shadowfactor and SVF"
    coords = SVF.coordheight(data)
    [svf,Shadowfactor] = SVF.reshape_SVF(data, coords,azimuth,elevation_angle,reshape=True,save_CSV=False,save_Im=False)
    "Now we need to separate the roof, wall and road SVF and SF"
    SVF_roof = (np.zeros([data.shape]) + np.ones(data>0))*svf
    SVF_road = (np.zeros([data.shape]) + np.ones(data==0))*svf

    """We evaluate the middle block only, but after the SVF and Shadowfactor are calculated"""
    [x_len,y_len] = data.shape
    data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]

    map_T_roof,map_T_wall,map_T_road, \
           capacities_roof,capacities_wall,capacities_road, \
           emissivity_roof,emissivity_wall,emissivity_road, \
           albedos_roof,albedos_wall,albedos_road, \
           lambdas_roof, lambdas_wall, lambdas_road, \
           T_inner_bc_roof, T_inner_bc_wall, T_inner_bc_road = initialize_map(Constants.layers,T_surf,data)

    map_T_old_roof = map_T_roof
    map_T_old_wall = map_T_wall
    map_T_old_road = map_T_road

    for t in range(time_steps):
        SW_dir = SW_down[t]/2
        SW_dif = SW_down[t]/2
        LW_d = LW_down[t]
        map_T_roof[:,:,0],map_T_wall[:,:,0],map_T_road[:,:,0] = surfacebalance(albedos_roof, albedos_wall, albedos_road, \
            emissivity_roof, emissivity_wall, emissivity_road, \
            capacities_roof, capacities_wall, capacities_road, \
            SVF_roof, SVF_wall, SVF_road, \
            WVF_wall, GVF_wall, \
            SF_roof, SF_wall, SF_road, \
            Constants.d_roof, Constants.d_wall, Constants.d_road, \
            lambdas_roof, lambdas_wall, lambdas_road, \
            map_T_old_roof[:,:,0], map_T_old_wall[:,:,0], map_T_old_road[:,:,0], \
            map_T_roof[:,:,1], map_T_wall[:,:,1], map_T_road[:,:,1], \
            delta_t, \
            Constants.sigma, \
            (1-elevation_angle), \
            SW_dif, SW_dir, \
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
        T_ave_roof[t] = np.mean(map_T_roof[:,:,0])
        T_ave_wall[t] = np.mean(map_T_wall[:,:,0])
        T_ave_road[t] = np.mean(map_T_road[:,:,0])

    return T_ave_roof, T_ave_wall, T_ave_road

"""PLOTFUNCTIONS"""
def PlotGreyMap(data,middle,v_max):
    plt.figure()
    if middle == True:
        [x_len,y_len] = data.shape
        plt.imshow(data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)],vmax=v_max)
    elif middle == False:
        plt.imshow(data,vmax=v_max)
    plt.show()

def PlotSurfaceTemp(T_ave_roof,T_ave_wall,T_ave_road, time_steps):
    time = np.ones([time_steps]) * Constants.delta_T

    plt.figure()
    plt.plot(time,T_ave_roof, label="roof")
    plt.plot(time,T_ave_wall, label="wall")
    plt.plot(time,T_ave_road, label="road")

    plt.xlabel("Time [s]")
    plt.ylabel("Average surface temperature [K]")
    plt.legend()
    plt.show()

