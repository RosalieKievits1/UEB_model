# Constants
import numpy as np

#stephan boltzman constant
sigma = 5.67e-8

# layers of simulation
layers_roof = 5
layers_road = 3
layers_wall = 5

# fractions of built environment in town:
# has to add up to one
# This can be expanded to contain more categories
# fraction of buildings in town [0-1]
building_frac = 0.25
# fraction of vegetation in town [0-1]
vegetation_frac = 0.25
# fraction of roads/pavement in town [0-1]
frac_roads = 1-building_frac-vegetation_frac

# layer thickness of road wall and roof [m]
d_roof = 0.005
d_wall = 0.1
d_fiber = 0.15
d_road = 0.1

#timestep
timestep = 2*60 #[s]
nr_of_steps = 1000

# albedos
a_bitumen = 0.08
a_asphalt = 0.12
a_grass = 0.3
a_glass = 0.21
a_brick = 0.3

# emissivities
e_asphalt = 0.88
e_grass = 0.98
e_bitumen = 0.95
e_grass = 0.956
e_glass = 0.9
e_brick = 0.9

# start temperatures
T_roof = 2+273.15
T_road = 5+273.15
T_grass = 10+273.15
T_ground = 10+273.15
# building heating temp
T_building = 20+273.15

# BUILDING MATERIALS
# heat capacities
# [J/m3K]
C_asphalt = 2251e3
C_bitumen = 2000e3
C_brick = 2018e3
C_fiber = 148e3

# thermal conductivities
# [W/mK]
lamb_asphalt = 0.75
lamb_grass = 1.1
lamb_bitumen = 0.8
lamb_brick = 1.31
lamb_fiber = 0.08

# Geometric properties:
# Hight over width ratio
H_W = 1
# roughness length [m], should be determined from AHN
z_0 = 1.2
# average building height
H = 20
# reference pressure
p_0 = 1e5 # [Pa]
R_d = 28.964917 # [g/mol]
C_pd = 1.005 # [kJ/kgK] heat capacity of dry air
rho_air = 1.2985 # [kg/m3] air density at first atmospheric level
# surface and first atmospheric level pressures (should these be inputs from DALES??)
p_surf = 1.01325e5 # [Pa] surface pressure
p_trop = 1.089e5 # [Pa] pressure at first atmospheric level (troposphere), according to literature ranges between 100 and 200mBar

#define the sky view factor
Phi_roof = 1
Phi_wall = 1/2*(H_W+1-np.sqrt(H_W**2+1))/H_W
Phi_road = np.sqrt(H_W**2+1)-H_W
# vector of sky view factors
Phi = [Phi_roof,Phi_wall,Phi_road]

"""For the solar position algorithm (based on Rotterdam)"""
latitude = 51.9
long_rd = 4.46
"Julian day and hour of the day in local mean time"
julianday = 285 #12 oct
hour = 11.5 # 11:30 am
