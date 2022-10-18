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

def lat_sens_fluxes(p_surf,p_atm,T_atm,T_can, T_road,T_wall,q_atm,U_atm,ave_height,C_drag,delta_z):

    T_a_corr = T_atm*exner(p_surf)/exner(p_atm)
    q_atm_corr = q_atm*q_sat(T_a_corr,p_surf)/q_sat(T_atm,p_atm)

    H_roof = Constants.C_pd*Constants.rho_air*(T_a_corr-T_can)/RES_roof
    LE_roof = Constants.L_v*Constants.rho_air*(q_atm_corr-q_atm)/RES_roof

    """Computing wind speeds"""
    W_can = np.sqrt(C_drag)*abs(U_atm)
    U_top = 2/np.pi * (np.log(ave_height/3/Constants.z_0)/np.log((delta_z + ave_height/3)/Constants.z_0)) * abs(U_atm)
    N = 0.5*Constants.H_W
    U_can = U_top*np.exp(-N/2)

    H_top = Constants.C_pd*Constants.rho_air*(T_a_corr-T_can)/RES_top
    LE_top = Constants.L_v*Constants.rho_air*(q_atm_corr-q_atm)/RES_top

    RES_road = 1/(11.8+4.2*np.sqrt(W_can**2+U_can**2))
    RES_wall = RES_road

    H_road = Constants.C_pd*Constants.rho_air*(T_road-T_can)/RES_road
    LE_road = Constants.L_v*Constants.rho_air*(q_sat(T_road,p_surf)-q_atm)/RES_road

    H_wall = Constants.C_pd*Constants.rho_air*(T_wall-T_can)/RES_wall
    LE_wall = 0

    return LE_roof,H_roof,LE_wall,H_wall,LE_road,H_road

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
                   delta_t,H_W):
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
    """define the sky view factor"""
    Phi_roof = 1
    Phi_wall = 1/2*(H_W+1-np.sqrt(H_W**2+1))/H_W
    Phi_road = np.sqrt(H_W**2+1)-H_W
    """vector of sky view factors"""
    SVF = [Phi_roof,Phi_wall,Phi_road]

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
    theta_zero = np.arcsin(min(1,(1/(H_W*np.tan(Zenith[t])))))
    SW_roof = SW_down[t]
    SW_wall = SW_down[t]*(1/H_W*(1/2-theta_zero/np.pi)+1/np.pi*np.tan(Zenith[t])*(1-np.cos(theta_zero)))
    SW_road = SW_down[t]*(2*theta_zero/np.pi-2/np.pi*H_W*np.tan(Zenith[t])*(1-np.cos(theta_zero)))

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

    """Latent and sensible heat fluxes"""
    #[LE_roof, LE_wall, LE_road, H_roof, H_wall, H_road] = lat_sens_fluxes(p_surf,Constants.p_atm,T_atm,T_can, map_temperatures_road[0,t-1],T_wall,q_atm,U_atm,ave_height,C_drag,delta_z):

    """ Net radiation"""
    netRad_roof = LW_net_roof + SW_net_roof - G_out_surf_roof #- LE_roof - H_roof
    netRad_wall = LW_net_wall + SW_net_wall - G_out_surf_wall #- LE_wall - H_wall
    netRad_road = LW_net_road + SW_net_road - G_out_surf_road #- LE_road - H_road

    """ Temperature change"""
    dT_roof = (netRad_roof/(roof_capacities[0]*roof_d[0]))*delta_t
    map_temperatures_roof[0,t] = map_temperatures_roof[0,t-1] + dT_roof

    dT_wall = (netRad_wall/(wall_capacities[0]*wall_d[0]))*delta_t
    map_temperatures_wall[0,t] = map_temperatures_wall[0,t-1] + dT_wall

    dT_road = (netRad_road/(road_capacities[0]*road_d[0]))*delta_t
    map_temperatures_road[0,t] = map_temperatures_road[0,t-1] + dT_road

    return map_temperatures_roof[0,t],map_temperatures_wall[0,t],map_temperatures_road[0,t]

"""function for temperature of each layer"""
def layer_balance_Masson(map_temperatures,layers,d,lambdas,t,T_inner_bc,delta_t,capacities,type):
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

def Masson_model(T_building,T_ground,T_air,nr_steps,H_W):
    """Intitialize all layers for all three surfacetypes"""
    [map_temperatures_roof,roof_d,roof_lambdas,roof_capacities] = initialize(Constants.layers_roof,nr_steps,T_air,T_building)
    [map_temperatures_wall,wall_d,wall_lambdas,wall_capacities] = initialize(Constants.layers_wall,nr_steps,T_air,T_building)
    [map_temperatures_road,road_d,road_lambdas,road_capacities] = initialize(Constants.layers_road,nr_steps,T_air,T_ground)

    """FOR THE ROOF"""
    roof_d[0] = Constants.d_roof
    roof_d[1:Constants.layers_roof-1] = Constants.d_wall
    roof_d[Constants.layers_roof-1] = Constants.d_fiber

    """initialize different materials for different layers"""
    roof_lambdas[0] = Constants.lamb_bitumen
    roof_capacities[0] = Constants.C_bitumen

    roof_lambdas[1:Constants.layers_roof-1]=Constants.lamb_brick
    roof_capacities[1:Constants.layers_roof-1]=Constants.C_brick

    roof_lambdas[Constants.layers_roof-1]=Constants.lamb_fiber
    roof_capacities[Constants.layers_roof-1]=Constants.C_fiber

    """FOR THE ROAD"""
    road_d[:] = Constants.d_road
    # initialize different materials for different layers
    road_lambdas[:] = Constants.lamb_asphalt
    road_capacities[:] = Constants.C_asphalt

    """FOR THE WALL"""
    wall_d[0:Constants.layers_wall-1] = Constants.d_wall
    wall_d[Constants.layers_wall-1] = Constants.d_fiber

    """initialize different materials for different layers"""
    wall_lambdas[0:Constants.layers_wall-1] = Constants.lamb_brick
    wall_capacities[0:Constants.layers_wall-1] = Constants.C_brick
    wall_lambdas[Constants.layers_wall-1]=Constants.lamb_fiber
    wall_capacities[Constants.layers_wall-1]=Constants.C_fiber


    """now we start with evolving over time"""
    for t in range(1,nr_steps):
        """Surface temperatures"""
        [map_temperatures_roof[0,t],map_temperatures_wall[0,t],map_temperatures_road[0,t]] = surfacebalance_Masson(Constants.albedos,Constants.emissivities,map_temperatures_roof,map_temperatures_wall,map_temperatures_road,\
                                                          Constants.sigma,t,roof_lambdas,roof_capacities,roof_d,\
                                                          wall_lambdas,wall_capacities,wall_d,\
                                                          road_lambdas,road_capacities,road_d, \
                                                          Constants.timestep,H_W)
        """Temperatures for each layer"""
        map_temperatures_roof[1:Constants.layers_roof,t] = layer_balance_Masson(map_temperatures_roof,Constants.layers_roof,roof_d,roof_lambdas,t,T_building,Constants.timestep,roof_capacities,type="roof")
        map_temperatures_wall[1:Constants.layers_wall,t] = layer_balance_Masson(map_temperatures_wall,Constants.layers_wall,wall_d,wall_lambdas,t,T_building,Constants.timestep,wall_capacities,type="wall")
        map_temperatures_road[1:Constants.layers_road,t] = layer_balance_Masson(map_temperatures_road,Constants.layers_road,wall_d,road_lambdas,t,T_ground,Constants.timestep,road_capacities,type="road")
    return map_temperatures_roof, map_temperatures_wall, map_temperatures_road

def Masson_Rotterdam(data, gridboxsize):
    [ave_height, delta, Roof_area, wall_area_total, Road_area,Water_area, Roof_frac, Wall_frac, Road_frac, Water_frac,H_W] = SVF.geometricProperties(data,gridboxsize)
    [map_temperatures_roof, map_temperatures_wall, map_temperatures_road] = Masson_model(Constants.T_building,Constants.T_ground,T_air,nr_steps,H_W)


"""Equations for map model"""
def initialize_map(layers,T_surf,T_inner_bc,data,emissivity_array,albedo_array):
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
    capacities = np.zeros([x_len,y_len,layers])
    lambdas = np.zeros([x_len,y_len,layers])
    d = np.zeros([x_len,y_len,layers])
    map_t = np.zeros([x_len,y_len,layers])
    T_inner_bc_mat = np.zeros([x_len,y_len])

    """Roofs"""
    lambdas[data>0,:] = Constants.lamb_bitumen
    d[data>0,:] = Constants.d_roof
    capacities[data>0,:] = Constants.C_bitumen
    lin_temp_roof = np.linspace(T_surf,T_inner_bc[0],layers)
    map_t[data>0,:] = lin_temp_roof
    T_inner_bc_mat[data>0] = T_inner_bc[0]

    """Roads"""
    lambdas[data==0,:] = Constants.lamb_asphalt
    d[data==0,:] = Constants.d_road
    capacities[data==0,:] = Constants.C_asphalt
    lin_temp_road = np.linspace(T_surf,T_inner_bc[2],layers)
    map_t[data==0,:] = lin_temp_road
    T_inner_bc_mat[data==0] = T_inner_bc[2]

    """Water"""
    lambdas[data==-1,:] = Constants.lamb_water
    d[data==-1,:] = Constants.d_water
    capacities[data==-1,:] = Constants.C_water
    lin_temp_water = np.linspace(T_surf,T_inner_bc[3],layers)
    map_t[data==-1,:] = lin_temp_water
    T_inner_bc_mat[data==-1] = T_inner_bc[3]

    """Albedos and emissivities"""
    emissivities = np.ndarray(data.shape)
    albedos = np.ndarray(data.shape)

    """Where buildings are placed"""
    emissivities[data>0]=emissivity_array[0]
    albedos[data>0]=albedo_array[0]
    """On ground areas"""
    emissivities[data==0]=emissivity_array[2]
    albedos[data==0]=albedo_array[2]
    """Water"""
    emissivities[data<0]=emissivity_array[3]
    albedos[data<0]=albedo_array[3]

    print(T_inner_bc_mat)
    return map_t,d,lambdas,capacities,T_inner_bc_mat,emissivities,albedos

def surfacebalance(albedos,emissivities,capacities,sigma,\
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

    """Longwave radiation"""
    LW_net = LW_down * SVF * emissivities - emissivities * map_temperatures_old**4 * sigma

    """ Short wave radiation"""
    SW_net = SW_dir * SVF * Shadowfactor * (1-albedos) + SW_diff * SVF * (1-albedos)

    """conduction"""
    lamb_ave_out_surf = (d[:,:,0]+d[:,:,1])/((d[:,:,0]/lambdas[:,:,0])+(d[:,:,1]/lambdas[:,:,1]))
    G_out_surf = lamb_ave_out_surf*((map_temperatures_old-map_temperatures_old_subs)/(1/2*(d[:,:,0]+d[:,:,1])))

    """ Net radiation"""
    netRad = LW_net + SW_net - G_out_surf

    #print("netrad = " + str(netRad))
    """ Temperature change"""
    dT = (netRad/(capacities[:,:,0]*d[:,:,0]))*delta_t
    map_temperatures = map_temperatures_old + dT

    return map_temperatures

def layer_balance(data, d, layers,lambdas,map_temperatures,map_temp_old,T_inner_bc_mat,capacities,delta_t):
    G_out = np.ndarray(data.shape)

    for l in range(1,layers):
        lamb_ave_in = (d[:,:,l-1]+d[:,:,l])/((d[:,:,l-1]/lambdas[:,:,l-1])+(d[:,:,l]/lambdas[:,:,l]))
        G_in = lamb_ave_in*((map_temp_old[:,:,l-1]-map_temp_old[:,:,l])/(1/2*(d[:,:,l-1]+d[:,:,l])))
        """for all layers before the last layer and after the first (surface) layer"""

        if (l < Constants.layers-1):
            lamb_ave_out = (d[:,:,l]+d[:,:,l+1])/((d[:,:,l]/lambdas[:,:,l])+(d[:,:,l+1]/lambdas[:,:,l+1]))
            """compute the convective fluxes in and out the layer"""
            G_out = lamb_ave_out*((map_temp_old[:,:,l]-map_temp_old[:,:,l+1])/(1/2*(d[:,:,l]+d[:,:,l+1])))
        if (l == layers-1):
            G_out = lambdas[:,:,l]*(map_temp_old[:,:,l]-T_inner_bc_mat)/(1/2*d[:,:,l])
            G_out[data<=0] = 0
        """Change in temperature"""
        dT = ((G_in-G_out)*delta_t)/(capacities[:,:,l]*d[:,:,l])
        map_temperatures[:,:,l] = map_temp_old[:,:,l] + dT
    return map_temperatures[:,:,1:layers]

def HeatEvolution(data,time_steps,delta_t):
    """There is no shadow and there are no obstructions, for now"""
    t_roof_ave = np.zeros(time_steps)
    t_ave = np.zeros(time_steps)
    [wallArea_matrix, wallArea_total] = SVF.wallArea(data)
    wall_layers = np.ndarray([wallArea_matrix.shape[0],wallArea_matrix.shape[1],Constants.layers])

    coords = SVF.coordheight(data)
    [svf,Shadowfactor,blocklength] = SVF.reshape_SVF(data,coords)
    """We evaluate the middle block only, but after the SVF and Shadowfactor are calculated"""
    [x_len,y_len] = data.shape
    data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]

    """Calculate and append the SVF"""
    #Shadowfactor = np.zeros(data.shape)
    #svf = np.ones(data.shape)

    map_t,d,lambdas,capacities,T_inner_bc_mat,albedos,emissivities = initialize_map(Constants.layers,T_2m[0],Constants.T_inner_bc,data,Constants.emissivities,Constants.albedos)

    map_t_old = map_t
    for t in range(time_steps):
        SW_dir = SW_down[t]/2
        SW_dif = SW_down[t]/2
        LW_d = LW_down[t]
        map_t[:,:,0] = surfacebalance(albedos, emissivities, capacities, Constants.sigma, \
                                      svf, Shadowfactor,SW_dir, SW_dif, LW_d, d, lambdas, delta_t, map_t_old[:, :, 0], map_t_old[:, :, 1])
        map_t[:,:,1:] = layer_balance(data, d, Constants.layers,lambdas,map_t,map_t_old,T_inner_bc_mat,capacities,delta_t)
        map_t_old = map_t
        t_ave[t] = np.mean(map_t[:,:,0])
        t_roof_ave[t] = np.mean(map_t[data>0,0])

    return t_roof_ave

"""PLOTFUNCTIONS"""
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

def plotTempComparison_Masson(map_temp_roof,map_temp_wall,map_temp_road,l):
    """
    :param map_temp: array of temperatures for each layer and timestep
    :param l: layer we want to compare
    :return: PLOT of temperatures for a specific layer for the three different surfaces
    """
    plt.figure()
    plt.xlabel("time")
    plt.ylabel("temperature [K]")
    plt.title("Temperature for layer " + str(l) + " of different surfaces")
    plt.plot(map_temp_roof[l,:],label= "Temperature for roof layer " + str(l) + " over time")
    plt.plot(map_temp_wall[l,:],label= "Temperature for wall layer " + str(l) + " over time")
    plt.plot(map_temp_road[l,:],label= "Temperature for road layer " + str(l) + " over time")
    plt.legend()
    plt.xlabel("time " + str(Constants.timestep) + "s")
    plt.ylabel("Temperature [K]")
    plt.show()
