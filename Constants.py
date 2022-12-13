# Constants
import numpy as np
import Sunpos

#stephan boltzman constant
sigma = 5.67e-8

# layers of simulation
layers = 5

"""layer thickness of road wall and roof [m]"""
d_roof = np.ones([layers]) * 0.05 #[0.05,0.1,0.15, 0.2] #
d_wall = np.ones([layers]) * 0.05 #[0.05,0.1,0.15, 0.2] #
d_road = np.ones([layers]) * 0.05 #[0.05,0.1,0.15, 0.2] #

"""timestep"""
timestep = 10*60 #[s]
nr_of_steps = 600

"""albedos"""
a_bitumen = 0.12
a_asphalt = 0.12
a_grass = 0.3
a_glass = 0.21
a_brick = 0.3
a_water = 0.009

"""emissivities"""
e_asphalt = 0.88
e_grass = 0.98
e_bitumen = 0.97
e_grass = 0.94 # from cabau
e_glass = 0.9
e_brick = 0.9
e_water = 0.95

"""vector of emissivities of roof wall and road"""
emissivities = [e_bitumen,e_brick,e_asphalt,e_water]
albedos = [a_bitumen,a_brick,a_asphalt,a_asphalt]

"""start temperatures"""
T_roof = 2+273.15
T_road = 5+273.15
T_grass = 10+273.15
T_ground = 10+273.15
T_water = 10+273.15
# building heating temp
T_building = 20+273.15

"""BUILDING MATERIALS"""
"""heat capacities"""
# [J/m3K]
C_asphalt = 2251e3
C_bitumen = 2000e3
C_brick = 2018e3
C_grass = 2000e3
C_fiber = 148e3
C_water = 4200e3
C_soil = 1900e3 # https://open.library.okstate.edu/rainorshine/chapter/13-2-soil-thermal-properties/

"""thermal conductivities"""
# [W/mK]
lamb_asphalt = 0.75
lamb_grass = 1.1
lamb_bitumen = 0.8
lamb_brick = 1.31
lamb_fiber = 0.08
lamb_water = 0.598
lamb_soil = 0.3

"""Geometric properties:"""
# Hight over width ratio
# H_W = 1
# roughness length [m], should be determined from AHN
z_0 = 1.2
# average building height
H = 20
# reference pressure
p_0 = 1e5 # [Pa]
R_d = 28.964917 # [g/mol]
C_pd = 1.005e3 # [J/kgK] heat capacity of dry air
rho_air = 1.2985 # [kg/m3] air density at first atmospheric level
# surface and first atmospheric level pressures (should these be inputs from DALES??)
p_atm = 1.01325e5 # [Pa] surface pressure
p_surf = 1.089e5 # [Pa] pressure at first atmospheric level (troposphere), according to literature ranges between 100 and 200mBar

"Aerodynamc resistances for roof and road"
res_roof = 1/0.03
res_road = 1/0.03


R_w = 461.52 # J/kgK gas constant of water
L_v = 2.5e6 #J/kg latent vaporization heat of water
T_0 = 273.15 # K ref temp
e_s_T0 = 0.611e3 #Pa e_s at reference temperature
eps = 0.622 # ratio of molar masses of vapor and dry air

"""For the solar position algorithm (based on Rotterdam)"""
latitude = 51.9
long_rd = 4.46
"Julian day and hour of the day in local mean time"
julianday = 305 #1 nov
hour = 10.5 # 10:30 am


