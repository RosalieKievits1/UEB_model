# import packages
import numpy as np
import pandas as pd
import Constants
import matplotlib.pyplot as plt
import SVF
from tqdm import tqdm
from numpy import random
import Sunpos
plt.rc('font', size=13) #for three plots togethes

"""Read in data"""
# data = pd.read_csv("cabauw_2018.csv", sep = ';')
# #data = pd.read_csv("era5_NL_2021.csv",sep=',')
# data.head()

# """upward sensible heat flux"""
# SHF = data.iloc[: , 32]
# """Relative humidity at 10 m"""
# q_first_layer = data.iloc[: , 30]
# """Upward Latent Heat flux"""
# LHF = data.iloc[: , 33]
# """upward longwave heat flux"""
# LW_up = data.iloc[: , 34]
# """downward longwave heat flux"""
# LW_down = data.iloc[: , 35]
# """upward shortwave heat flux"""
# SW_up = data.iloc[: , 36]
# """downward shortwave heat flux"""
# SW_down = data.iloc[: , 37]
# """solar zenith angle"""
# Zenith = data.iloc[: ,38]*np.pi/180
# Zenith = data.iloc[: ,6]*np.pi/180
# GHI = data.iloc[: ,2]
# DNI = data.iloc[:,5]
# DHI = data.iloc[:,4]
# D_perc = DNI*np.cos(Zenith)/GHI
# """the temperature at 2 m high (use as ic for surface temp)"""
# T_2m = data.iloc[: ,24]
# T_air = T_2m[0]
# """Surface pressure"""
# p_surf = data.iloc[: ,5]
#

# plt.figure()
# #print(np.polyfit(Zenith, D_perc, deg=3))
# #plt.plot(Zenith,0.1318+1.0317*Zenith-0.6612*Zenith**2,'r')
# # plt.plot(Zenith,0.8-0.36*Zenith**2,'r')
# plt.scatter(Zenith,SW_down,marker='x')
# #plt.plot(Zenith,np.cos(Zenith),'r')
# # plt.xlim((0,np.pi/2))
# plt.ylim((0,np.max(SW_down)))
#
# plt.xlabel('Zenith Angle [Rad]')
# plt.ylabel('Shortwave Radiation Flux [W/m2]')
# plt.show()



def exner(pressure):
    p_zero = 10e5
    return (pressure/p_zero)**(Constants.R_d/Constants.C_pd)

def q_sat(T,p):
    "e_sat is the saturation vapor pressure, determined by the Clausius-Clapeyron equation"
    "T must be in Kelvin and this returns the e_sat in Pa, and q in kg/kg"
    e_sat = Constants.e_s_T0 * np.log((Constants.L_v/Constants.R_w*(1/Constants.T_0-1/T+0.0001)))
    # if T==Constants.T_0:
    #     e_sat = Constants.e_s_T0
    q_sat = (Constants.eps*e_sat)/(p - (1-Constants.eps)*e_sat)
    return q_sat

def T_pot(T,p):
    p_zero = 10e5
    T_pot = T * (p_zero/p)**0.286
    return T_pot


"""Equations for map model"""
def initialize_map(layers,shape):
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
    [x_len,y_len] = shape
    capacities_roof = np.ones([x_len,y_len,layers]) * Constants.C_brick
    capacities_wall = np.ones([x_len,y_len,layers]) * Constants.C_brick
    capacities_road = np.ones([x_len,y_len,layers]) * Constants.C_soil
    lambdas_roof = np.ones([x_len,y_len,layers]) * Constants.lamb_brick
    lambdas_wall = np.ones([x_len,y_len,layers]) * Constants.lamb_brick
    lambdas_road = np.ones([x_len,y_len,layers]) * Constants.lamb_soil
    T_inner_bc_roof = np.ones([x_len,y_len]) * Constants.T_building
    T_inner_bc_wall = np.ones([x_len,y_len]) * Constants.T_building
    map_T_roof = np.ndarray([x_len,y_len,layers])
    map_T_wall = np.ndarray([x_len,y_len,layers])
    map_T_road = np.ndarray([x_len,y_len,layers])
    map_T_water = np.ndarray([x_len,y_len,layers])
    map_T_ground = np.ndarray([x_len,y_len,layers])

    """Roofs"""
    map_T_roof[:,:,0:layers] = Constants.T_air #lin_temp_roof
    capacities_roof[:,:,0] = Constants.C_bitumen
    lambdas_roof[:,:,0] = Constants.lamb_bitumen

    """walls"""
    map_T_wall[:,:,0:layers] = Constants.T_air #lin_temp_wall

    """Roads"""
    map_T_road[:,:,0:layers] = Constants.T_air
    map_T_water[:,:,0:layers] = Constants.T_air
    capacities_road[:,:,0] = Constants.C_asphalt
    lambdas_road[:,:,0] = Constants.lamb_asphalt

    """Albedos and emissivities"""
    emissivity_roof = np.ones([x_len,y_len]) * Constants.e_bitumen
    emissivity_wall = np.ones([x_len,y_len]) * Constants.e_brick
    emissivity_road = np.ones([x_len,y_len]) * Constants.e_asphalt

    albedos_roof = np.ones([x_len,y_len]) * Constants.a_bitumen
    albedos_wall = np.ones([x_len,y_len]) * Constants.a_brick
    albedos_road = np.ones([x_len,y_len]) * Constants.a_asphalt

    return map_T_roof,map_T_wall,map_T_road,map_T_water,map_T_ground, \
           capacities_roof,capacities_wall,capacities_road, \
           emissivity_roof,emissivity_wall,emissivity_road, \
           albedos_roof,albedos_wall,albedos_road, \
           lambdas_roof, lambdas_wall, lambdas_road, \
           T_inner_bc_roof, T_inner_bc_wall

def surfacebalance(albedos_roof, albedos_wall, albedos_road,
                   emissivities_roof, emissivities_wall, emissivities_road,
                   capacities_roof, capacities_wall, capacities_road,
                   SVF_roof, SVF_wall, SVF_road,
                   SF_roof, SF_wall, SF_road,
                   d_roof, d_wall, d_road,d_water,
                   lambdas_roof, lambdas_wall, lambdas_road,
                   T_old_roof, T_old_wall, T_old_road,T_old_water,
                   T_old_subs_roof, T_old_subs_wall, T_old_subs_road,T_old_subs_water,
                   road_frac,water_frac,
                   delta_t,
                   sigma,theta_z,
                   SW_diff, SW_dir,    # from dales
                   T_firstlayer,q_first_layer,
                   LW_down, res_roof, res_wall, res_road):
    """
    Returns a map of the surface temperatures for all three surface types
    """
    WVF_roof = 1-SVF_roof
    WVF_road = 1-SVF_road
    GVF_wall = SVF_wall
    RVF_wall = 0#WVF_roof/2
    WVF_wall = 1-SVF_wall-GVF_wall#-RVF_wall

    "Emissivity and Albedo of ground surface, for reflections"
    emissivities_ground = emissivities_road*road_frac+Constants.e_water*water_frac
    albedos_ground = albedos_road*road_frac+Constants.a_water*water_frac
    T_old_ground = T_old_road*road_frac+T_old_water*water_frac

    """Longwave radiation"""
    # LW_net_roof = LW_down * emissivities_roof * SVF_roof - emissivities_roof * T_old_roof**4 * sigma + \
    #               (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_roof * emissivities_roof
    # LW_net_wall = LW_down * emissivities_wall * SVF_wall - emissivities_wall * T_old_wall**4 * sigma + \
    #               (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_wall * emissivities_wall + \
    #               (LW_down * SVF_road * (1-emissivities_ground) + emissivities_ground * T_old_ground**4 * sigma) * GVF_wall * emissivities_wall + \
    #               (LW_down * SVF_roof * (1-emissivities_roof) + emissivities_roof * T_old_roof**4 * sigma) * RVF_wall * emissivities_wall
    # LW_net_road = LW_down * emissivities_road * SVF_road - emissivities_road * T_old_road**4 * sigma + \
    #               (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_road * emissivities_road
    # LW_net_water = LW_down * Constants.e_water * SVF_road - Constants.e_water * T_old_water**4 * sigma + \
    #               (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_road * Constants.e_water
    # LW_net_roof = LW_down * emissivities_roof * SVF_roof - emissivities_roof * T_old_roof**4 * sigma + \
    #              (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_roof * emissivities_roof

    "SPLITTED"
    "LW Roof Splitted"
    LW_roof_first = LW_down * emissivities_roof * SVF_roof
    LW_roof_em = emissivities_roof * T_old_roof**4 * sigma
    LW_roof_w = (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_roof * emissivities_roof
    LW_net_roof = LW_roof_first - LW_roof_em + LW_roof_w

    "LW Wall Splitted"
    LW_w_first = LW_down * emissivities_wall * SVF_wall
    LW_w_em = emissivities_wall * T_old_wall**4 * sigma
    LW_w_wall =  (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_wall * emissivities_wall
    LW_w_road = (LW_down * SVF_road * (1-emissivities_ground) + emissivities_ground * T_old_ground**4 * sigma) * GVF_wall * emissivities_wall
    LW_w_roof = (LW_down * SVF_roof * (1-emissivities_roof) + emissivities_roof * T_old_roof**4 * sigma) * RVF_wall * emissivities_wall
    LW_net_wall = LW_w_first - LW_w_em + LW_w_wall + LW_w_roof + LW_w_road

    "LW road splitted"
    LW_road_first = LW_down * emissivities_road * SVF_road
    LW_road_em = emissivities_road * T_old_road**4 * sigma
    LW_road_w = (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_road * emissivities_road
    LW_net_road = LW_road_first - LW_road_em + LW_road_w

    "LW water Splitted"
    LW_water_first = LW_down * Constants.e_water * SVF_road
    LW_water_em = Constants.e_water * T_old_water**4 * sigma
    LW_water_w = (LW_down * SVF_wall * (1-emissivities_wall) + emissivities_wall * T_old_wall**4 * sigma) * WVF_road * Constants.e_water
    LW_net_water = LW_water_first - LW_water_em + LW_water_w

    "Ground Surfaces"
    LW_ground_w = road_frac*LW_road_w + water_frac*LW_water_w
    LW_ground_first = road_frac*LW_road_first + water_frac*LW_water_first
    LW_ground_em = road_frac*LW_road_em + water_frac*LW_water_em
    LW_ground_net = road_frac*LW_net_road + water_frac*LW_net_water

    "SHORTWAVE Radiation"
    # SW_net_roof = SW_dir * SF_roof * (1-albedos_roof) + SW_diff * SVF_roof * (1-albedos_roof) + \
    #               (SW_dir * SF_wall * np.tan(theta_z) + SW_diff * SVF_wall) * albedos_wall * (1-albedos_roof) * WVF_roof
    # SW_net_wall = SW_dir * SF_wall * (1-albedos_wall) * np.tan(theta_z) + SW_diff * SVF_wall * (1-albedos_wall) + \
    #               (SW_dir * SF_wall * np.tan(theta_z) + SW_diff * SVF_wall) * albedos_wall * (1-albedos_wall) * WVF_wall + \
    #               (SW_dir * SF_road + SW_diff * SVF_road) * albedos_ground * (1-albedos_wall) * GVF_wall + \
    #               (SW_dir * SF_roof + SW_diff * SVF_roof) * albedos_roof * (1-albedos_wall) * RVF_wall
    # SW_net_road = SW_dir * SF_road * (1-albedos_road) + SW_diff * SVF_road * (1-albedos_road) + \
    #               (SW_dir * SF_wall * np.tan(theta_z) + SW_diff * SVF_wall) * albedos_wall * (1-albedos_road) * (1-SVF_road)
    # SW_net_water = SW_dir * SF_road * (1-Constants.a_water) + SW_diff * SVF_road * (1-Constants.a_water) + \
    #               (SW_dir * SF_wall * np.tan(theta_z) + SW_diff * SVF_wall) * albedos_wall * (1-Constants.a_water) * (1-SVF_road)

    "SPLIT"
    "SW roof split"
    SW_roof_dir = SW_dir * SF_roof * (1-albedos_roof)
    SW_roof_dif = SW_diff * SVF_roof * (1-albedos_roof)
    SW_roof_w = (SW_dir * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_roof) * WVF_roof
    SW_net_roof = SW_roof_dir + SW_roof_dif + SW_roof_w

    "SW wall split"
    SW_w_dir = SW_dir * SF_wall * (1-albedos_wall)
    SW_w_dif = SW_diff * SVF_wall * (1-albedos_wall)
    SW_w_wall = (SW_dir * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_wall) * WVF_wall
    SW_w_road = (SW_dir * SF_road + SW_diff * SVF_road) * albedos_ground * (1-albedos_wall) * GVF_wall
    SW_w_roof = (SW_dir * SF_roof + SW_diff * SVF_roof) * albedos_roof * (1-albedos_wall) * RVF_wall
    SW_net_wall = SW_w_dir + SW_w_dif + SW_w_wall + SW_w_road + SW_w_roof

    "SW road split"
    SW_road_dir = SW_dir * SF_road * (1-albedos_road)
    SW_road_dif = SW_diff * SVF_road * (1-albedos_road)
    SW_road_w = (SW_dir * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-albedos_road) * (1-SVF_road)
    SW_net_road = SW_road_dif + SW_road_dir + SW_road_w

    "SW Water split"
    SW_water_dir = SW_dir * SF_road * (1-Constants.a_water)
    SW_water_dif = SW_diff * SVF_road * (1-Constants.a_water)
    SW_water_w = (SW_dir * SF_wall + SW_diff * SVF_wall) * albedos_wall * (1-Constants.a_water) * (1-SVF_road)
    SW_net_water = SW_water_dir + SW_water_dif + SW_water_w

    "Ground Surfaces"
    SW_ground_w = road_frac*SW_road_w + water_frac*SW_water_w
    SW_ground_dif = road_frac*SW_road_dif + water_frac*SW_water_dif
    SW_ground_dir = road_frac*SW_road_dir + water_frac*SW_water_dir
    SW_ground_net = road_frac*SW_net_road + water_frac*SW_net_water

    " Latent and Sensible Heat fluxes "
    SHF_roof = Constants.C_pd * Constants.rho_air * (1/res_roof) * (T_pot(T_old_roof,Constants.p_atm) - T_pot(T_firstlayer,Constants.p_atm))
    SHF_wall = Constants.C_pd * Constants.rho_air * (1/res_wall) * (T_pot(T_old_wall,Constants.p_atm) - T_pot(T_firstlayer,Constants.p_atm))
    SHF_road = Constants.C_pd * Constants.rho_air * (1/res_road) * (T_pot(T_old_road,Constants.p_atm) - T_pot(T_firstlayer,Constants.p_atm))
    SHF_water = Constants.C_pd * Constants.rho_air * (1/Constants.res_water) * (T_pot(T_old_water,Constants.p_atm) - T_pot(T_firstlayer,Constants.p_atm))

    LHF_roof = 0
    LHF_wall = 0
    LHF_road = 0
    LHF_water = Constants.L_v * Constants.rho_air * (1/Constants.res_water) * (q_sat(T_old_water,Constants.p_atm) - q_sat(T_firstlayer,Constants.p_atm))

    "SHF LHF Ground"
    SHF_ground = road_frac*SHF_road + water_frac*SHF_water
    LHF_ground = road_frac*LHF_road + water_frac*LHF_water


    " conduction"
    lamb_ave_out_surf_roof = (d_roof[0]+d_roof[1])/((d_roof[0]/lambdas_roof[:,:,0])+(d_roof[1]/lambdas_roof[:,:,1]))
    G_out_surf_roof = lamb_ave_out_surf_roof*((T_old_roof-T_old_subs_roof)/(1/2*(d_roof[0]+d_road[1])))
    lamb_ave_out_surf_wall = (d_wall[0]+d_wall[1])/((d_wall[0]/lambdas_wall[:,:,0])+(d_wall[1]/lambdas_wall[:,:,1]))
    G_out_surf_wall = lamb_ave_out_surf_wall*((T_old_wall-T_old_subs_wall)/(1/2*(d_wall[0]+d_wall[1])))
    lamb_ave_out_surf_road = (d_road[0]+d_road[1])/((d_road[0]/lambdas_road[:,:,0])+(d_road[1]/lambdas_road[:,:,1]))
    G_out_surf_road = lamb_ave_out_surf_road*((T_old_road-T_old_subs_road)/(1/2*(d_road[0]+d_road[1])))
    "For water the lambdas are all the same"
    G_out_surf_water = Constants.lamb_water*((T_old_water-T_old_subs_water)/(1/2*(d_water[0]+d_water[1])))
    G_out_surf_ground = road_frac*G_out_surf_road + water_frac*G_out_surf_water

    " Net radiation "
    netRad_roof = LW_net_roof + SW_net_roof - G_out_surf_roof - SHF_roof - LHF_roof
    netRad_wall = LW_net_wall + SW_net_wall - G_out_surf_wall - SHF_wall - LHF_wall
    netRad_road = LW_net_road + SW_net_road - G_out_surf_road - SHF_road - LHF_road
    netRad_water = LW_net_water + SW_net_water - G_out_surf_water - SHF_water - LHF_water

    " Temperature change "
    dT_roof = (netRad_roof/(capacities_roof[:,:,0]*d_roof[0]))*delta_t
    map_T_roof = T_old_roof + dT_roof
    dT_wall = (netRad_wall/(capacities_wall[:,:,0]*d_wall[0]))*delta_t
    map_T_wall = T_old_wall + dT_wall
    dT_road = (netRad_road/(capacities_road[:,:,0]*d_road[0]))*delta_t
    map_T_road = T_old_road + dT_road
    dT_water = (netRad_water/(Constants.C_water*d_water[0]))*delta_t
    map_T_water = T_old_water + dT_water

    map_T_ground = water_frac*map_T_water+road_frac*map_T_road

    return map_T_roof,map_T_wall,map_T_road,map_T_water, map_T_ground, \
           LW_net_roof, SW_net_roof, G_out_surf_roof, SHF_roof, \
           LW_net_wall, SW_net_wall, G_out_surf_wall, SHF_wall, \
           LW_net_road, SW_net_road, G_out_surf_road, SHF_road, \
           LW_net_water, SW_net_water, G_out_surf_water, SHF_water, LHF_water, \
           SW_ground_net, LW_ground_net, SHF_ground, LHF_ground, G_out_surf_ground, \
           SW_roof_dif,SW_roof_dir,SW_roof_w, \
           SW_ground_dif,SW_ground_dir,SW_ground_w, \
           SW_w_dif,SW_w_dir,SW_w_wall,SW_w_roof,SW_w_road, \
           LW_roof_first, LW_roof_em, LW_roof_w, \
           LW_w_first, LW_w_em, LW_w_wall, LW_w_roof, LW_w_road, \
           LW_ground_first, LW_ground_em, LW_ground_w

def layer_balance(d_roof, d_wall, d_road,d_water,
                  lambdas_roof, lambdas_wall, lambdas_road,
                  map_T_roof, map_T_wall, map_T_road,map_T_water,map_T_ground,
                  T_old_roof, T_old_wall, T_old_road,T_old_water,
                  T_inner_bc_roof, T_inner_bc_wall,
                  capacities_roof, capacities_wall, capacities_road,
                  water_frac,road_frac,
                  layers, delta_t):

    "Returns the temperature change in all layers beneath the surface layer"
    for l in range(1,layers):
        lamb_ave_in_roof = (d_roof[l]+d_roof[l-1])/((d_roof[l-1]/lambdas_roof[:,:,l-1])+(d_roof[l]/lambdas_roof[:,:,l]))
        G_in_roof = lamb_ave_in_roof*((T_old_roof[:,:,l-1]-T_old_roof[:,:,l])/(1/2*(d_roof[l-1]+d_roof[l])))
        lamb_ave_in_wall = (d_wall[l]+d_wall[l-1])/((d_wall[l-1]/lambdas_wall[:,:,l-1])+(d_wall[l]/lambdas_wall[:,:,l]))
        G_in_wall = lamb_ave_in_wall*((T_old_wall[:,:,l-1]-T_old_wall[:,:,l])/(1/2*(d_wall[l-1]+d_wall[l])))
        lamb_ave_in_road = (d_road[l]+d_road[l-1])/((d_road[l-1]/lambdas_road[:,:,l-1])+(d_road[l]/lambdas_road[:,:,l]))
        G_in_road = lamb_ave_in_road*((T_old_road[:,:,l-1]-T_old_road[:,:,l])/(1/2*(d_road[l-1]+d_road[l])))
        "For water the lambdas are all the same"
        G_in_water = Constants.lamb_water*((T_old_water[:,:,l-1]-T_old_water[:,:,l])/(1/2*(d_water[l-1]+d_water[l])))

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
            "For water the lambdas are all the same"
            G_out_water = Constants.lamb_water*((T_old_water[:,:,l]-T_old_water[:,:,l+1])/(1/2*(d_water[l]+d_water[l+1])))

        if (l == layers-1):
            G_out_roof = lambdas_roof[:,:,l]*(T_old_roof[:,:,l]-T_inner_bc_roof)/(1/2*d_roof[l])
            G_out_wall = lambdas_wall[:,:,l]*(T_old_wall[:,:,l]-T_inner_bc_wall)/(1/2*d_wall[l])
            G_out_road = 0
            G_out_water = 0
        """Change in temperature"""
        dT_roof = ((G_in_roof-G_out_roof)*delta_t)/(capacities_roof[:,:,l]*d_roof[l])
        dT_wall = ((G_in_wall-G_out_wall)*delta_t)/(capacities_wall[:,:,l]*d_wall[l])
        dT_road = ((G_in_road-G_out_road)*delta_t)/(capacities_road[:,:,l]*d_road[l])
        dT_water = ((G_in_water-G_out_water)*delta_t)/(Constants.C_water*d_water[l])

        "New temperatures"
        map_T_roof[:,:,l] = T_old_roof[:,:,l] + dT_roof
        map_T_wall[:,:,l] = T_old_wall[:,:,l] + dT_wall
        map_T_road[:,:,l] = T_old_road[:,:,l] + dT_road
        map_T_water[:,:,l] = T_old_water[:,:,l] + dT_water

        map_T_ground[:,:,l] = water_frac*map_T_water[:,:,l]+road_frac*map_T_road[:,:,l]

    return map_T_roof[:,:,1:layers], map_T_wall[:,:,1:layers], map_T_road[:,:,1:layers],map_T_water[:,:,1:layers],map_T_ground[:,:,1:layers]

def HeatEvolution(time_steps,delta_t,SW_down,Zenith,LW_down,T_2m,q_first_layer,
                  SVF_roof,SVF_wall,SVF_road,
                  SF_roof,SF_wall,SF_road,
                  Roof_frac, Road_frac, Wall_frac, Water_frac,
                  res_roof, res_wall, res_road):

    "Arrays that store average surface temperatures"
    T_ave_roof = np.empty((time_steps,Constants.layers))
    T_ave_wall = np.empty((time_steps,Constants.layers))
    T_ave_road = np.empty((time_steps,Constants.layers))
    T_ave_water = np.empty((time_steps,Constants.layers))
    T_ave_ground = np.empty((time_steps,Constants.layers))
    T_ave_surf = np.empty((time_steps))

    "Arrays to store all surface averaged heat fluxes"
    LW_ave_roof = np.empty((time_steps))
    LW_ave_wall = np.empty((time_steps))
    LW_ave_road = np.empty((time_steps))
    LW_ave_water = np.empty((time_steps))
    LW_ave_ground = np.empty((time_steps))

    SW_wall_dif = np.empty((time_steps))
    SW_wall_dir = np.empty((time_steps))
    SW_wall_wall = np.empty((time_steps))
    SW_wall_roof = np.empty((time_steps))
    SW_wall_road = np.empty((time_steps))

    SW_r_dir = np.empty((time_steps))
    SW_r_dif = np.empty((time_steps))
    SW_r_wall = np.empty((time_steps))

    SW_g_dir = np.empty((time_steps))
    SW_g_dif = np.empty((time_steps))
    SW_g_wall = np.empty((time_steps))

    LW_w_first = np.empty((time_steps))
    LW_w_em = np.empty((time_steps))
    LW_w_wall = np.empty((time_steps))
    LW_w_roof = np.empty((time_steps))
    LW_w_road = np.empty((time_steps))

    LW_r_first = np.empty((time_steps))
    LW_r_em = np.empty((time_steps))
    LW_r_wall = np.empty((time_steps))

    LW_g_first = np.empty((time_steps))
    LW_g_em = np.empty((time_steps))
    LW_g_wall = np.empty((time_steps))

    SW_ave_roof = np.empty((time_steps))
    SW_ave_wall = np.empty((time_steps))
    SW_ave_road = np.empty((time_steps))
    SW_ave_water = np.empty((time_steps))
    SW_ave_ground = np.empty((time_steps))

    G_ave_roof = np.empty((time_steps))
    G_ave_wall = np.empty((time_steps))
    G_ave_road = np.empty((time_steps))
    G_ave_water = np.empty((time_steps))
    G_ave_ground = np.empty((time_steps))

    SHF_ave_roof = np.empty((time_steps))
    SHF_ave_road = np.empty((time_steps))
    SHF_ave_wall = np.empty((time_steps))
    SHF_ave_water = np.empty((time_steps))
    LHF_ave_water = np.empty((time_steps))

    SHF_ave_ground = np.empty((time_steps))
    LHF_ave_ground = np.empty((time_steps))

    "Now we need to separate the roof, wall and road SVF and SF"
    [x_len,y_len] = SVF_roof.shape

    map_T_roof,map_T_wall,map_T_road,map_T_water,map_T_ground, \
           capacities_roof,capacities_wall,capacities_road, \
           emissivity_roof,emissivity_wall,emissivity_road, \
           albedos_roof,albedos_wall,albedos_road, \
           lambdas_roof, lambdas_wall, lambdas_road, \
           T_inner_bc_roof, T_inner_bc_wall = initialize_map(Constants.layers,[x_len,y_len])

    map_T_old_roof = map_T_roof
    map_T_old_wall = map_T_wall
    map_T_old_road = map_T_road
    map_T_old_water = map_T_water

    "The water and road fractions are fractions of the entire surface area"
    Water_frac_new = Water_frac/(Water_frac+Road_frac)
    Road_frac_new = Road_frac/(Water_frac+Road_frac)
    Road_frac_new = np.nan_to_num(Road_frac_new, nan=0) #nan=np.nanmean(Road_frac_new))
    Water_frac_new = np.nan_to_num(Water_frac_new,nan=0) #nan=np.nanmean(Water_frac_new))
    for t in tqdm(range(time_steps)):

        SW_dir = SW_down[t]*np.cos(Zenith[t])
        SW_dif = SW_down[t]*(1-np.cos(Zenith[t]))
        LW_d = LW_down[t]
        "Set these to constants for now"
        T_firstlayer = T_2m[t]
        q_firstlayer = q_first_layer[t]
        [map_T_roof[:,:,0],map_T_wall[:,:,0],map_T_road[:,:,0],map_T_water[:,:,0],map_T_ground[:,:,0],\
         LW_net_roof, SW_net_roof, G_out_surf_roof, SHF_roof, \
           LW_net_wall, SW_net_wall, G_out_surf_wall, SHF_wall, \
           LW_net_road, SW_net_road, G_out_surf_road, SHF_road, \
           LW_net_water, SW_net_water, G_out_surf_water, SHF_water, LHF_water, \
            SW_ground_net, LW_ground_net, SHF_ground, LHF_ground, G_out_surf_ground, \
            SW_roof_dif,SW_roof_dir,SW_roof_w, \
            SW_road_dif,SW_road_dir,SW_road_w, \
            SW_w_dif,SW_w_dir,SW_w_w,SW_w_roof,SW_w_road,\
            LW_roof_first, LW_roof_em, LW_roof_w, \
            LW_wall_first, LW_wall_em, LW_wall_wall, LW_wall_roof, LW_wall_road, \
            LW_ground_first, LW_ground_em, LW_ground_w] = \
                surfacebalance(albedos_roof, albedos_wall, albedos_road, \
            emissivity_roof, emissivity_wall, emissivity_road, \
            capacities_roof, capacities_wall, capacities_road, \
            SVF_roof, SVF_wall, SVF_road, \
            SF_roof[t,:,:], SF_wall[t,:,:], SF_road[t,:,:], \
            Constants.d_roof, Constants.d_wall, Constants.d_road,Constants.d_water, \
            lambdas_roof, lambdas_wall, lambdas_road, \
            map_T_old_roof[:,:,0], map_T_old_wall[:,:,0], map_T_old_road[:,:,0], map_T_old_water[:,:,0],\
            map_T_roof[:,:,1], map_T_wall[:,:,1], map_T_road[:,:,1],map_T_water[:,:,1], \
            Road_frac_new, Water_frac_new, \
            delta_t, \
            Constants.sigma,Zenith[t], \
            SW_dif, SW_dir, \
            T_firstlayer, q_firstlayer,\
            LW_d, res_roof, res_wall, res_road)

        map_T_roof[:,:,1:], map_T_wall[:,:,1:], map_T_road[:,:,1:], map_T_water[:,:,1:],map_T_ground[:,:,1:] = \
            layer_balance(Constants.d_roof, Constants.d_wall, Constants.d_road,Constants.d_water,
                  lambdas_roof, lambdas_wall, lambdas_road,
                  map_T_roof, map_T_wall, map_T_road,map_T_water,map_T_ground,
                  map_T_old_roof, map_T_old_wall, map_T_old_road,map_T_old_water,
                  T_inner_bc_roof, T_inner_bc_wall,
                  capacities_roof, capacities_wall, capacities_road,
                  Water_frac_new,Road_frac_new,
                  Constants.layers, delta_t)

        map_T_old_roof = map_T_roof
        map_T_old_wall = map_T_wall
        map_T_old_road = map_T_road
        map_T_old_water = map_T_water

        "Average surface temperatures"
        for l in range(Constants.layers):
            T_ave_roof[t,l] = np.mean(map_T_roof[:,:,l])
            T_ave_wall[t,l] = np.mean(map_T_wall[:,:,l])
            T_ave_road[t,l] = np.mean(map_T_road[:,:,l])
            T_ave_water[t,l] = np.mean(map_T_water[:,:,l])
            T_ave_ground[t,l] = np.mean(map_T_ground[:,:,l])

        "Average surface Temperature"
        T_ave_surf[t] = np.mean(map_T_roof[:,:,0]*Roof_frac + map_T_wall[:,:,0]*Wall_frac + map_T_road[:,:,0]*Road_frac + map_T_water[:,:,0]*Water_frac)

        "Surface averaged Fluxes"
        LW_ave_roof[t] = np.mean(LW_net_roof)
        LW_ave_wall[t] = np.mean(LW_net_wall)
        LW_ave_road[t] = np.mean(LW_net_road)
        LW_ave_water[t] = np.mean(LW_net_water)
        LW_ave_ground[t] = np.mean(LW_ground_net)

        SW_ave_roof[t] = np.mean(SW_net_roof)
        SW_ave_wall[t] = np.mean(SW_net_wall)
        SW_ave_road[t] = np.mean(SW_net_road)
        SW_ave_water[t] = np.mean(SW_net_water)
        SW_ave_ground[t] = np.mean(SW_ground_net)

        LHF_ave_water[t] = np.mean(LHF_water)
        G_ave_roof[t] = np.mean(G_out_surf_roof)
        G_ave_wall[t] = np.mean(G_out_surf_wall)
        G_ave_road[t] = np.mean(G_out_surf_road)
        G_ave_water[t] = np.mean(G_out_surf_water)
        G_ave_ground[t] = np.mean(G_out_surf_ground)

        SHF_ave_roof[t] = np.mean(SHF_roof)
        SHF_ave_road[t] = np.mean(SHF_road)
        SHF_ave_wall[t] = np.mean(SHF_wall)
        SHF_ave_water[t] = np.mean(SHF_water)

        SHF_ave_ground[t] = np.mean(SHF_ground)
        LHF_ave_ground[t] = np.mean(LHF_ground)

        "SW fluxes splitted"
        SW_r_dif[t] = np.mean(SW_roof_dif)
        SW_r_dir[t] = np.mean(SW_roof_dir)
        SW_r_wall[t] = np.mean(SW_roof_w)

        SW_g_dif[t] = np.mean(SW_road_dif)
        SW_g_dir[t] = np.mean(SW_road_dir)
        SW_g_wall[t] = np.mean(SW_road_w)

        SW_wall_dif[t] = np.mean(SW_w_dif)
        SW_wall_dir[t] = np.mean(SW_w_dir)
        SW_wall_wall[t] = np.mean(SW_w_w)
        SW_wall_roof[t] = np.mean(SW_w_roof)
        SW_wall_road[t] = np.mean(SW_w_road)

        "LW fluxes Splitted"
        LW_r_first[t] = np.mean(LW_roof_first)
        LW_r_em[t] = np.mean(LW_roof_em)
        LW_r_wall[t] = np.mean(LW_roof_w)

        LW_w_first[t] = np.mean(LW_wall_first)
        LW_w_em[t] = np.mean(LW_wall_em)
        LW_w_wall[t] = np.mean(LW_wall_wall)
        LW_w_roof[t] = np.mean(LW_wall_roof)
        LW_w_road[t] = np.mean(LW_wall_road)

        LW_g_first[t] = np.mean(LW_ground_first)
        LW_g_em[t] = np.mean(LW_ground_em)
        LW_g_wall[t] = np.mean(LW_ground_w)

    return T_ave_roof, T_ave_wall, T_ave_road,T_ave_water,T_ave_ground,T_ave_surf, \
           LW_ave_roof, SW_ave_roof, G_ave_roof, SHF_ave_roof, \
           LW_ave_wall, SW_ave_wall, G_ave_wall, SHF_ave_wall, \
           LW_ave_road, SW_ave_road, G_ave_road, SHF_ave_road, \
           LW_ave_water, SW_ave_water, G_ave_water, SHF_ave_water, LHF_ave_water, \
           SW_ave_ground, LW_ave_ground, SHF_ave_ground, LHF_ave_ground, G_ave_ground, \
           SW_r_dir,SW_r_dif,SW_r_wall, \
           SW_g_dir,SW_g_dif,SW_g_wall, \
           SW_wall_dir, SW_wall_dif, SW_wall_wall,SW_wall_roof,SW_wall_road,\
           LW_r_first, LW_r_em, LW_r_wall, \
           LW_w_first, LW_w_em, LW_w_wall, LW_w_roof, LW_w_road, \
           LW_g_first, LW_g_em, LW_g_wall

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

def PlotSurfaceTemp(T_ave_roof,T_ave_wall,T_ave_road,T_ave_water, T_ave_ground,T_2m,T_ave_surf,time_steps,show=False):
    time = (np.arange(time_steps)* Constants.timestep/3600)
    plt.ylim((Constants.T_air-Constants.T_air_amp*2,Constants.T_air+Constants.T_air_amp*2))
    plt.figure()
    plt.plot(time,T_2m, 'blue', label="Air Temp")
    plt.plot(time,T_ave_roof[:,0], label="Roof")
    plt.plot(time,T_ave_wall[:,0], label="Wall")
    #plt.plot(time,T_ave_road[:,0], label="road")
    #plt.plot(time,T_ave_water[:,0], label="water")
    plt.plot(time,T_ave_ground[:,0], label="ground")
    #plt.plot(time,T_ave_surf, label="Slab Surface")
    plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
    plt.xlabel("Time [h]")
    plt.ylabel("Surface Temperature [K]")
    plt.legend(loc='upper right')
    if show == True:
        plt.show()
    return

def PlotTempLayers(T_ave,T_2m,time_steps,show=False):
    time = (np.arange(time_steps)* Constants.timestep/3600)
    plt.figure()
    plt.plot(time,T_2m, label="Air Temp")
    for l in range(Constants.layers):
        plt.plot(time,T_ave[:,l], label=str(l))
    plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
    plt.xlabel("Time [h]")
    plt.ylabel("Temperature [K]")
    plt.legend(loc='upper right')
    if show == True:
        plt.show()
    return

def PlotSurfaceFluxes(nr_of_steps,Fluxes,Flux_names,show=False):
    time = (np.arange(nr_of_steps)*Constants.timestep / 3600)
    plt.figure()
    for i in range(len(Fluxes)):
        plt.plot(time,Fluxes[i], label=Flux_names[i])
    plt.rcParams['font.family'] = ['Comic Sans', 'sans-serif']
    plt.xlabel("Time [h]")
    plt.ylabel("Flux [W/m2K]")
    plt.legend(loc='upper right')
    if show == True:
        plt.show()
    return
