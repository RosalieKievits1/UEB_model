# import packages
import numpy as np
import pandas as pd
import Constants
import SVF
import matplotlib.pyplot as plt
import Functions

# This script is for simulating an urban energy balance, taking into consideration the town geometry.
# 1: first we define all governing equations and test the surface energy and temperature for a flat surface
# 2: Next we implement a surface with height differences (start with infinite canyon)
# 3: Implement solar angle and shading effects
# 4: implement reflection effects
# 5: implement snow and rain surface obstruction
# 6: implement human heat sources

nr_steps = Functions.nr_steps
T_air = Functions.T_air

# """MASSON"""
# """Intitialize all layers for all three surfacetypes"""
# [map_temperatures_roof,roof_d,roof_lambdas,roof_capacities] = Functions.initialize(Constants.layers_roof,nr_steps,T_air,Constants.T_building)
# [map_temperatures_road,road_d,road_lambdas,road_capacities] = Functions.initialize(Constants.layers_wall,nr_steps,T_air,Constants.T_ground)
# [map_temperatures_wall,wall_d,wall_lambdas,wall_capacities] = Functions.initialize(Constants.layers_road,nr_steps,T_air,Constants.T_building)
#
# """FOR THE ROOF"""
# roof_d[0] = Constants.d_roof
# roof_d[1:Constants.layers_roof-1] = Constants.d_wall
# roof_d[Constants.layers_roof-1] = Constants.d_fiber
#
# """initialize different materials for different layers"""
# roof_lambdas[0] = Constants.lamb_bitumen
# roof_capacities[0] = Constants.C_bitumen
#
# roof_lambdas[1:Constants.layers_roof-1]=Constants.lamb_brick
# roof_capacities[1:Constants.layers_roof-1]=Constants.C_brick
#
# roof_lambdas[Constants.layers_roof-1]=Constants.lamb_fiber
# roof_capacities[Constants.layers_roof-1]=Constants.C_fiber
#
# """FOR THE ROAD"""
# road_d[:] = Constants.d_road
# # initialize different materials for different layers
# road_lambdas[:] = Constants.lamb_asphalt
# road_capacities[:] = Constants.C_asphalt
#
# """FOR THE WALL"""
# wall_d[0:Constants.layers_wall-1] = Constants.d_wall
# wall_d[Constants.layers_wall-1] = Constants.d_fiber
#
# """initialize different materials for different layers"""
# wall_lambdas[0:Constants.layers_wall-1] = Constants.lamb_brick
# wall_capacities[0:Constants.layers_wall-1] = Constants.C_brick
# wall_lambdas[Constants.layers_wall-1]=Constants.lamb_fiber
# wall_capacities[Constants.layers_wall-1]=Constants.C_fiber


"""now we start with evolving over time"""
# for t in range(1,nr_steps):
#     """Surface temperatures"""
#     [map_temperatures_roof[0,t],map_temperatures_wall[0,t],map_temperatures_road[0,t]] = Functions.surfacebalance_Masson(albedos,emissivities,map_temperatures_roof,map_temperatures_wall,map_temperatures_road,\
#                                                           Constants.sigma,t,roof_lambdas,roof_capacities,roof_d,\
#                                                           wall_lambdas,wall_capacities,wall_d,\
#                                                           road_lambdas,road_capacities,road_d, \
#                                                           Constants.timestep,Constants.Phi)
#     """Temperatures for each layer"""
#     map_temperatures_roof[1:Constants.layers_roof,t] = Functions.layer_balance_Masson(map_temperatures_roof,Constants.layers_roof,roof_d,roof_lambdas,t,Constants.T_building,Constants.timestep,roof_capacities,type="roof")
#     map_temperatures_wall[1:Constants.layers_wall,t] = Functions.layer_balance_Masson(map_temperatures_wall,Constants.layers_wall,wall_d,wall_lambdas,t,Constants.T_building,Constants.timestep,wall_capacities,type="wall")
#     map_temperatures_road[1:Constants.layers_road,t] = Functions.layer_balance_Masson(map_temperatures_road,Constants.layers_road,wall_d,road_lambdas,t,Constants.T_ground,Constants.timestep,road_capacities,type="road")


"""Plotting of the layers temperatures for each material"""
#Functions.plotTemp_Masson(map_temperatures_roof)
#Functions.plotTemp(map_temperatures_road)
#Functions.plotTemp(map_temperatures_wall)


"""City using SVF and shadow casting"""
data = SVF.datasquare(SVF.dtm1,SVF.dsm1,SVF.dtm2,SVF.dsm2,SVF.dtm3,SVF.dsm3,SVF.dtm4,SVF.dsm4)
coords = SVF.coordheight(data)
SVFs = SVF.calc_SVF(coords, SVF.steps_psi , SVF.steps_beta,SVF.max_radius,SVF.blocklength)
SFs = SVF.shadowfactor(coords, 286,Constants.latitude,Constants.long_rd,10.5,SVF.steps_beta,SVF.blocklength)

"""Reshape the shadowfactors and SVF back to nd array"""
SVF_matrix = np.ndarray([data.shape[0]/2,data.shape[1]/2])
SF_matrix = np.ndarray([data.shape[0]/2,data.shape[1]/2])
for i in range(SVF.blocklength):
    SVF_matrix[coords[i,0],coords[i,1]] = SVFs[i]
    SF_matrix[coords[i,0],coords[i,1]] = SFs[i]

print(Functions.HeatEvolution(data,nr_steps,Constants.timestep,SVF_matrix,SF_matrix))

