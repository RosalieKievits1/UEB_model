# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt

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
nr_steps = np.size(LW_up,0)


T_air = T_2m[0]

def exner(pressure,t):
    return (pressure/p_surf[t])**(Constants.R_d/Constants.C_pd)

def initialize(layers,nr_steps,T_surf,T_inner_bc):
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
    capacities = np.zeros(layers)
    lambdas = np.zeros(layers)
    d = np.zeros(layers)
    map_t = np.zeros((layers,nr_steps))
    lin_temp = np.linspace(T_surf,T_inner_bc,layers)
    for l in range(layers):
        map_t[l,0] = lin_temp[l]
    return map_t,d,lambdas,capacities

""" Function for surface balance"""
def surfacebalance_Masson(albedos,emissivities,map_temperatures_roof,map_temperatures_wall,map_temperatures_road,\
                   sigma,t,roof_lambdas,roof_capacities,roof_d,\
                   wall_lambdas,wall_capacities,wall_d,\
                   road_lambdas,road_capacities,road_d,\
                   delta_t,SVF):
    """
    :param albedo: albedo of surface material
    :param emissivity: emissivity of surface material
    :param map_temperatures: ndarray of amount of layers and timesteps
    :param sigma: stephan boltzman constant
    :param t: current timestep
    :param lambdas: array of heat conductivities (lambdas) for each layer
    :param capacities: array of volumetric heat capacities for each layer
    :param d: array of thicknesses in m
    :param delta_t: timestep size
    :return: surface temperature for timestep t
    """
    """Longwave radiation including longwave trapping"""
    LW_net_roof = emissivities[0] * SVF[0] * LW_down[t] - emissivities[0] * map_temperatures_roof[0,t-1]**4 * sigma
    LW_net_wall = emissivities[1] * SVF[1] * LW_down[t] \
             - emissivities[1] * map_temperatures_wall[0,t-1]**4 * sigma \
             + emissivities[1]*SVF[1]*sigma*emissivities[2]*map_temperatures_road[0,t-1]**4 \
             + emissivities[1]**2*(1-2*SVF[1])*sigma*map_temperatures_wall[0,t-1]**4 \
             + emissivities[1]*(1-emissivities[2])*SVF[1]*SVF[2]*LW_down[t] \
             + emissivities[1]*(1-emissivities[1])*SVF[1]*(1-2*SVF[1])*LW_down[t] \
             + emissivities[1]**2*(1-emissivities[1])*(1-2*SVF[1])**2*sigma*map_temperatures_wall[0,t-1]**4 \
             + emissivities[1]**2*(1-emissivities[2])*SVF[1]*(1-SVF[2])*sigma*map_temperatures_wall[0,t-1]**4 \
             + emissivities[1]*(1-emissivities[1])*SVF[1]*(1-2*SVF[1])*sigma*emissivities[2]*map_temperatures_road[0,t-1]**4
    LW_net_road = emissivities[2] * SVF[2] * LW_down[t] \
             - emissivities[2] * map_temperatures_road[0,t-1]**4 * sigma \
             + emissivities[1]*emissivities[2]*sigma*(1-SVF[2])*map_temperatures_wall[0,t-1]**4 \
             + emissivities[2]*(1-emissivities[1])*(1-SVF[2])*(SVF[1])*LW_down[t] \
             + emissivities[2]*emissivities[1]*(1-emissivities[2])*(1-SVF[2])*(1-2*SVF[1])*sigma*map_temperatures_wall[0,t-1]**4 \
             + emissivities[2]*(1-emissivities[1])*(1-SVF[2])*SVF[1]*sigma*emissivities[2]*map_temperatures_road[0,t-1]**4

    """ Short wave radiation"""
    """compute the radiation based on solar zenith angle"""
    theta_zero = np.arcsin(min(1,(1/(Constants.H_W*np.tan(Zenith[t])))))
    SW_roof = SW_down[t]
    SW_wall = SW_down[t]*(1/Constants.H_W*(1/2-theta_zero/np.pi)+1/np.pi*np.tan(Zenith[t])*(1-np.cos(theta_zero)))
    SW_road = SW_down[t]*(2*theta_zero/np.pi-2/np.pi*Constants.H_W*np.tan(Zenith[t])*(1-np.cos(theta_zero)))

    SW_net_roof = SW_roof*(1-albedos[0])

    """M is the sum of reflections between roof and wall"""
    M_wall = (albedos[1]*SW_wall+SVF[1]*albedos[1]*albedos[2]*SW_road)/ \
             (1-(1-2*SVF[1])*albedos[1]+(1-SVF[2])*SVF[1]*albedos[2]*albedos[1])
    M_road = (albedos[2]*SW_road+(1-SVF[2])*albedos[2]*(albedos[1]*SW_wall+SVF[1]*albedos[1]*albedos[2]*SW_road))/ \
             (1-(1-2*SVF[1])*albedos[1]+(1-SVF[2])*SVF[1]*albedos[2]*albedos[1])
    SW_net_wall = (1-albedos[1]) * SW_wall + (1-albedos[1])*(1-2*SVF[1])* M_wall + (1-albedos[1])*SVF[1]*M_road
    SW_net_road = (1-albedos[2]) * SW_road + (1-albedos[2])*(1-SVF[2])* M_wall

    """conduction"""
    lamb_ave_out_surf_roof = (roof_d[0]+roof_d[1])/((roof_d[0]/roof_lambdas[0])+(roof_d[1]/roof_lambdas[1]))
    G_out_surf_roof = lamb_ave_out_surf_roof*((map_temperatures_roof[0,t-1]-map_temperatures_roof[1,t-1])/(1/2*(roof_d[0]+roof_d[1])))

    lamb_ave_out_surf_wall = (wall_d[0]+wall_d[1])/((wall_d[0]/wall_lambdas[0])+(wall_d[1]/wall_lambdas[1]))
    G_out_surf_wall = lamb_ave_out_surf_wall*((map_temperatures_wall[0,t-1]-map_temperatures_wall[1,t-1])/(1/2*(wall_d[0]+wall_d[1])))

    lamb_ave_out_surf_road = (road_d[0]+road_d[1])/((road_d[0]/road_lambdas[0])+(road_d[1]/road_lambdas[1]))
    G_out_surf_road = lamb_ave_out_surf_road*((map_temperatures_road[0,t-1]-map_temperatures_road[1,t-1])/(1/2*(road_d[0]+road_d[1])))

    """ Net radiation"""
    netRad_roof = LW_net_roof + SW_net_roof - G_out_surf_roof
    netRad_wall = LW_net_wall + SW_net_wall - G_out_surf_wall
    netRad_road = LW_net_road + SW_net_road - G_out_surf_road

    """ Temperature change"""
    dT_roof = (netRad_roof/(roof_capacities[0]*roof_d[0]))*delta_t
    map_temperatures_roof[0,t] = map_temperatures_roof[0,t-1] + dT_roof

    dT_wall = (netRad_wall/(wall_capacities[0]*wall_d[0]))*delta_t
    map_temperatures_wall[0,t] = map_temperatures_wall[0,t-1] + dT_wall

    dT_road = (netRad_road/(road_capacities[0]*road_d[0]))*delta_t
    map_temperatures_road[0,t] = map_temperatures_road[0,t-1] + dT_road

    return map_temperatures_roof[0,t],map_temperatures_wall[0,t],map_temperatures_road[0,t]

"""function for temperature of each layer"""
def layer_balance(map_temperatures,layers,d,lambdas,t,T_inner_bc,delta_t,capacities,type):
    """
    :param map_temperatures: ndarray of amount of layers and timesteps
    :param layers: amount of layers for this surface type
    :param d: array of thickness
    :param lambdas: array of heat conductivities (lambda) for each layer
    :param t: Current timestep
    :param T_inner_bc: Boundary concition for last layer
    :param delta_t: timestep
    :param capacities: array of volumetric heat capacities
    :return: temperatures for timestep t for all but the surface layer
    """
    for l in range(1,layers):
        lamb_ave_in = (d[l-1]+d[l])/((d[l-1]/lambdas[l-1])+(d[l]/lambdas[l]))
        G_in = lamb_ave_in*((map_temperatures[l-1,t-1]-map_temperatures[l,t-1])/(1/2*(d[l-1]+d[l])))
        # for all layers before the last layer and after the first (surface) layer
        if (l == layers-1):
            if (type=="road"):
                G_out = 0
            else:
                G_out = lambdas[l]*(map_temperatures[l,t-1]-T_inner_bc)/(1/2*d[l])
        elif (l < Constants.layers_roof-1):
            lamb_ave_out = (d[l]+d[l+1])/((d[l]/lambdas[l])+(d[l+1]/lambdas[l+1]))
            # compute the convective fluxes in and out the layer
            G_out = lamb_ave_out*((map_temperatures[l,t-1]-map_temperatures[l+1,t-1])/(1/2*(d[l]+d[l+1])))

        # Change in temperature
        dT = ((G_in-G_out)*delta_t)/(capacities[l]*d[l])
        map_temperatures[l,t] = map_temperatures[l,t-1] + dT
    return map_temperatures[1:layers,t]


def surfacebalance(albedo_array,emissivity_array,capacities,sigma,\
                   SVF,Shadowfactor,SW_diff,SW_dir,LW_down,d,lambdas,delta_t,map_temperatures_old,map_temperatures_old_subs):
    """
    :param albedo_array: array with albedos for roof road and wall
    :param emissivity_array: array with emissivities for roof road and wall
    :param capacities: array with capacities for roof road and wall, for each layer
    :param sigma: stephan boltsman constant
    :param SVF: size (data) matrix of SVF (0-1) for each point
    :param Shadowfactor: size (data) matrix of SVF [0,1] for each point
    :param SW_diff: diffuse solar radiation
    :param SW_dir: direct solar radiation
    :param LW_down: longwave downwelling radiation
    :param d: array of thicknesses for each layer
    :param lambdas: ndarrat of size data with lambda for each point and layer
    :param delta_t: timestep
    :param map_temperatures_old: old temperatures for the surface layer
    :param map_temperatures_old_subs: old temperatures for the layer below the surface layer
    :return: New temperatures for the surface layer
    """

    emissivities = np.ndarray(data.shape)
    albedos = np.ndarray(data.shape)
    """Where buildings are placed"""
    emissivities[data[data>0]]=emissivity_array[0]
    albedos[data[data>0]]=albedo_array[0]
    """On ground areas"""
    emissivity_array[data[data==0]]=emissivity_array[2]
    albedos[data[data==0]]=albedo_array[2]

    """Longwave radiation"""
    LW_net = LW_down * SVF * emissivities - emissivities * map_temperatures_old**4 * sigma

    """ Short wave radiation"""
    SW_net = SW_dir * SVF * Shadowfactor * (1-albedos) + SW_diff * SVF * (1-albedos)

    """conduction"""
    lamb_ave_out_surf = (d[0]+d[1])/((d[0]/lambdas[0])+(d[1]/lambdas[1]))
    G_out_surf = lamb_ave_out_surf*((map_temperatures_old-map_temperatures_old_subs)/(1/2*(d[0]+d[1])))

    """ Net radiation"""
    netRad = LW_net + SW_net - G_out_surf

    """ Temperature change"""
    dT = (netRad/(capacities[:,:,0]*d[0]))*delta_t
    map_temperatures = map_temperatures_old + dT

    return map_temperatures


"""Plotfunctions"""
"""PLOT TEMPERATURES"""
def plotTemp_Masson(map_temp):
    """
    :param map_temp: array of temperatures for each layer and timestep
    :return: PLOT of temperatures
    """
    plt.figure()
    plt.xlabel("time")
    plt.ylabel("temperature [K]")
    plt.title("Temperature for different layers")
    for l in range(np.size(map_temp,0)):
        plt.plot(map_temp[l,:],label= "Temperature for layer " + str(l))
    plt.legend()
    plt.xlabel("time " + str(Constants.timestep) + "s")
    plt.ylabel("Temperature [K]")
    plt.show()
