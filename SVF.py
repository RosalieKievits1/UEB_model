import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
from tqdm import tqdm
import time
import sys

#data_excel = tf.imread('M5_37FZ1.TIF')  # dtm (topography)
import Constants
import Sunpos

"""Now we want to calculate the sky view factor"""
steps_beta = 180 # so we range in steps of 2 degrees
steps_psi = 90 # so we range in steps of 2 degrees
max_radius = 500 # max radius is 500 m
"""define the gridboxsize of the model"""
gridboxsize = 5
"""objects below 1 m we do not look at"""
minheight = 1

"""DSM's and DTM's"""
"""we need 4 databoxes to account for the SVF's on the side"""
# linksboven
dtm1 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/M5_37EZ2.TIF'
dsm1 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/R5_37EZ2.TIF'
# rechtsboven
dtm2 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/M5_37FZ1.TIF'
dsm2 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/R5_37FZ1.TIF'
# linksonder
dtm3 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/M5_37GN2.TIF'
dsm3 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/R5_37GN2.TIF'
# rechtsonder
dtm4 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/M5_37HN1.TIF'
dsm4 = '/Users/rosaliekievits/Desktop/Tiff bestanden MEP/R5_37HN1.TIF'


def readdata(minheight,dsm,dtm):
    """dsm (all info, with building)"""
    data = tf.imread(dsm)
    """dtm (topography)"""
    data_topo = tf.imread(dtm)
    """remove extreme large numbers for water and set to zero."""
    data_water = np.zeros(data.shape)
    data_water[data > 10 ** 38] = 1
    data[data > 10 ** 38] = 0
    data_topo[data_topo > 10 ** 38] = 0

    data_diff = data - data_topo
    """round to 2 decimals"""
    data_diff = np.around(data_diff, 3)

    """If all surrounding tiles are zero the middle one might be a mistake of just a lantern or something"""
    """But we only do this if we use the 0.5 m data"""
    #[x_len, y_len] = np.shape(data)
    # for i in range(x_len):  # maybe inefficient??
    #     for j in range(y_len):
    #         if data_diff[i,j] != 0 and i < (x_len -1) and j < (y_len-1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j+1] ==0 and data_diff[i+1,j-1] ==0 and data_diff[i-1,j+1] == 0 and data_diff[i-1,j-1] ==0:
    #             data_diff[i, j] = 0
    #         elif data_diff[i,j] != 0 and i == 0 and j < (y_len-1) and data_diff[i+1,j] == 0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j+1] ==0 and data_diff[i+1,j-1] ==0:
    #             data_diff[i,j] = 0
    #         elif data_diff[i,j] != 0 and i == x_len-1 and j < (y_len-1) and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i-1,j+1] == 0 and data_diff[i-1,j-1] ==0:
    #             data_diff[i, j] = 0
    #         elif data_diff[i,j] != 0 and j == 0 and i < (x_len-1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i+1,j+1] ==0 and data_diff[i-1,j+1] == 0:
    #             data_diff[i, j] = 0
    #         elif data_diff[i,j] != 0 and j == (y_len-1) and i < (x_len -1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j-1] ==0 and data_diff[i-1,j-1] ==0:
    #             data_diff[i, j] = 0

    """filter all heights below the min height out"""
    datadiffcopy = data_diff
    datadiffcopy[data_diff<minheight] = 0
    data_final = datadiffcopy
    """All water elements are set to zero"""
    data_final = data_final - data_water
    return data_final

def datasquare(dtm1,dsm1,dtm2,dsm2,dtm3,dsm3,dtm4,dsm4):
    """
    We need to glue four boxes together such that we can evaluate the square box in the middle
    """
    block1 = readdata(minheight,dsm1,dtm1)
    block2 = readdata(minheight,dsm2,dtm2)
    block3 = readdata(minheight,dsm3,dtm3)
    block4 = readdata(minheight,dsm4,dtm4)
    [x_len,y_len] = np.shape(block1)
    """now we make a block four times the size of the blocks"""
    bigblock = np.ndarray([2*x_len,2*y_len])
    """left upper block"""
    bigblock[:x_len,:y_len] = block1
    """right upper block"""
    bigblock[:x_len,y_len::] = block2
    """left lower block"""
    bigblock[x_len::,:y_len] = block3
    """right lower block"""
    bigblock[x_len::,y_len::] = block4
    return bigblock

"""First we store the data in a more workable form"""
def coordheight(data):
    """
    create an array with 3 columns for x, y, and z for each tile
    :param data: the data array with the height for each tile
    :return: 3 columns for x, y, and z
    """
    """From here on we set the height of the water elements back to 0"""
    data[data<0] = 0
    [x_len,y_len] = np.shape(data)
    coords = np.ndarray([x_len*y_len,3])
    """ so we start with the list of coordinates with all the points we want to evaluate
    all other points are after that, for this we use 2 different counters."""
    rowcount_block = int((x_len/2)*(y_len/2))
    rowcount_center = 0
    """we need to make a list of coordinates where the center block is first"""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if ((x_len/4)<=i and i<(3*x_len/4) and (y_len/4)<=j and j<(3*y_len/4)):
                coords[rowcount_center,0] = i
                coords[rowcount_center,1] = j
                coords[rowcount_center,2] = data[i,j]
                rowcount_center += 1
            elif (i<(x_len/4) or i>=(3*x_len/4) or j<(y_len/4) or j>=(3*y_len/4)):
                coords[rowcount_block,0] = i
                coords[rowcount_block,1] = j
                coords[rowcount_block,2] = data[i,j]
                rowcount_block += 1
    return coords

def dist(point, coord):
    """
    :param point: evaluation point (x,y,z)
    :param coord: array of coordinates with heights
    :return: the distance from each coordinate to the point and the angle
    """
    # Columns is dx
    dx = (coord[1]-point[1])*gridboxsize
    # Rows is dy
    dy = (coord[0]-point[0])*gridboxsize
    dist = np.sqrt(abs(dx)**2 + abs(dy)**2)
    """angle is 0 north direction"""
    angle = np.arctan2(dy,dx)+np.pi/2
    return dist,angle

def dist_round(point, coord,steps_beta):
    """
    :param point: evaluation point (x,y,z)
    :param coord: array of coordinates with heights
    :return: the distance from each coordinate to the point and the angle
    """
    dx = (coord[1] - point[1])*gridboxsize
    dy = (coord[0] - point[0])*gridboxsize
    dist = np.around(np.sqrt(abs(dx)**2 + abs(dy)**2),3)
    angle = np.around(np.arctan(dx/dy),5)
    #betas = np.linspace(0,2*np.pi,steps_beta+1)
    #angle_rounded = betas[(abs(betas-angle)).argmin()]
    return dist,angle

def dome(point, coords, maxR):
    """
    :param point: point we are evaluating
    :param coords: array of coordinates with heights
    :param maxR: maximum radius in which we think the coordinates can influence the SVF
    :return: a dome of points that we take into account to evaluate the SVF
    """
    angle = np.ndarray([coords.shape[0],1])
    radius = np.ndarray([coords.shape[0],1])
    for i in range(coords.shape[0]):
        radius[i],angle[i] = dist(point,coords[i,:])
    coords = np.concatenate([coords,radius],axis=1)
    coords = np.concatenate([coords,angle],axis=1)
    """the dome consist of points higher than the view height and within the radius we want"""
    dome = coords[(np.logical_and(coords[:,3]<maxR,coords[:,3]>0.1)),:]
    dome = dome[(dome[:,2]>point[2]),:]
    return dome

def d_area(radius, d_beta,height,maxR):
    """we assume the dome is large, and d_theta and d_phi small enough,
     such that we can approximate the d_area as a flat surface"""
    dx = radius*2*np.sin(d_beta/2)
    #dx = gridboxsize
    """multiply by maxR over radius such that the small area gets projected on the dome"""
    d_area = (dx*height)*maxR/radius
    return d_area

# def calc_SVF(coords, steps_psi , steps_beta,max_radius,blocklength):
#     """
#     Function to calculate the sky view factor.
#     We create a dome around a point with a certain radius,
#     and for eacht point in the dome we evaluate of the height of this point blocks the view
#     :param coords: all coordinates of our dataset
#     :param steps_phi: the steps in phi (between 0 and 2pi)
#     :param steps_theta: the steps in theta (between 0 and 2pi)
#     :param max_radius: maximum radius we think influences the svf
#     :param blocklength: the first amount of points in our data set we want to evaluate
#     :return: SVF for all points
#     """
#     """Vertical (phi) and horizontal (theta) angle on the dome"""
#     psi_lin = np.linspace(0,np.pi,steps_psi)
#     betas = np.zeros(steps_beta)
#     d_beta = 2*np.pi/steps_beta
#     d_psi = np.pi/steps_psi
#     betas_lin = np.linspace(0,2*np.pi,steps_beta+1)
#     SVF = np.ndarray([blocklength,1])
#
#     for i in tqdm(range(blocklength),desc="loop over points"):
#         point = coords[i,:]
#         """ we throw away all point outside the dome
#         # dome is now a 5 column array of points:
#         # the 5 columns: x,y,z,radius,angle beta"""
#         dome_p = dome(point, coords, max_radius)
#         """we loop over all points inside the dome"""
#         for d in tqdm(range(dome_p.shape[0]),desc="dome loop"):
#             for p in range(psi_lin.shape[0]):
#                 """evaluate if the height of the point is higher than the height of the point"""
#                 h_psi = np.tan(psi_lin[p])*(dome_p[d,3]) #
#                 heightdif = dome_p[d,2]-point[2]
#                 if (heightdif>=h_psi):
#                     """ if there already exist a blocking for that angle beta:
#                     calculate of the area of this blocking is more"""
#                     area = d_area(dome_p[d,3],d_beta,heightdif,max_radius)
#                     angle_rounded = betas[(abs(betas_lin-dome_p[d,4])).argmin()]
#                     if (betas[angle_rounded]<area):
#                         betas[angle_rounded] = area
#         """this is the analytical dome area but should make same assumption as for d_area
#         dome_area = max_radius**2*2*np.pi"""
#         dome_area = (max_radius*np.sin(d_psi/2)*2)*(max_radius*np.sin(d_beta/2)*2)*steps_beta*steps_psi
#         """The SVF is the fraction of area of the dome that is not blocked"""
#         SVF[i] = (dome_area - np.sum(betas,0))/dome_area
#     return SVF

def calc_SVF(coords, steps_psi , steps_beta,max_radius,blocklength):
    """
    Function to calculate the sky view factor.
    We create a dome around a point with a certain radius,
    and for eacht point in the dome we evaluate of the height of this point blocks the view
    :param coords: all coordinates of our dataset
    :param steps_phi: the steps in phi (between 0 and 2pi)
    :param steps_theta: the steps in theta (between 0 and 2pi)
    :param max_radius: maximum radius we think influences the svf
    :param blocklength: the first amount of points in our data set we want to evaluate
    :return: SVF for all points
    """
    """Vertical (phi) and horizontal (theta) angle on the dome"""
    d_beta = 2*np.pi/steps_beta
    d_psi = np.pi/steps_psi
    betas_lin = np.linspace(0,2*np.pi,steps_beta)
    SVF = np.ndarray([blocklength,1])

    for i in tqdm(range(blocklength),desc="loop over points"):
        point = coords[i,:]
        """ we throw away all point outside the dome
        # dome is now a 5 column array of points:
        # the 5 columns: x,y,z,radius,angle theta"""
        dome_p = dome(point, coords, max_radius)
        betas = np.zeros(steps_beta)
        """we loop over theta"""
        print(dome_p.shape)
        for d in tqdm(range(dome_p.shape[0]),desc="dome loop"):
            heightdif = dome_p[d,2]-point[2]
            area = d_area(dome_p[d,3],d_beta,heightdif,max_radius)

            """The angles of the min and max angle of the building"""
            beta_min = - np.arcsin(gridboxsize/2/dome_p[d,3]) + dome_p[d,4]
            beta_max = np.arcsin(gridboxsize/2/dome_p[d,3]) + dome_p[d,4]

            """Where the index of betas fall within the min and max beta, and there is not already a larger blocking"""
            betas[np.nonzero(betas[np.logical_and((beta_min<=betas_lin), (betas_lin<beta_max))]<area)] = area


        """this is the analytical dome area but should make same assumption as for d_area
        dome_area = max_radius**2*2*np.pi"""
        dome_area = (max_radius*np.sin(d_psi/2)*2)*(max_radius*np.sin(d_beta/2)*2)*steps_beta*steps_psi
        """The SVF is the fraction of area of the dome that is not blocked"""
        print(betas)
        SVF[i] = (dome_area - np.sum(betas))/dome_area
        print(SVF[i])
    return SVF

def shadowfactor(coords, julianday,latitude,longitude,LMT,steps_beta,blocklength):
    """
    :param coords: all other points, x,y,z values
    :param julianday: julian day of the year
    :param latitude: latitude of location
    :param longitude: longitude of location
    :param LMT: local mean time
    :param steps_theta: amount of steps in horizontal dome angle
    :param blocklength: these are all the points in coords
        that are in the data we want to evaluate
    :return: the shadowfactor of that point:
    """
    [azimuth,elevation_angle] = Sunpos.solarpos(julianday,latitude,longitude,LMT)
    Shadowfactor = np.ndarray([blocklength,1])
    d_beta = 2*np.pi/steps_beta

    for i in tqdm(range(blocklength),desc="loop over points in block for Shadowfactor"):
        """the point we are currently evaluating"""
        point = coords[i,:]

        for j in range(coords.shape[0]):
            radius, angle = dist(point,coords[j])
            """if the angle is within a very small range as the angle of the sun"""
            """The angles of the min and max angle of the building"""
            beta_min = - np.arcsin(gridboxsize/2/radius) + angle
            beta_max = np.arcsin(gridboxsize/2/radius) + angle

            if (beta_min <= azimuth and azimuth <= beta_max):
                """if the elevation angle times the radius is smaller than the height of that point
                the shadowfactor is zero since that point blocks the sun"""
                if ((np.tan(elevation_angle)*radius)<(coords[j,2]-point[2])):
                    Shadowfactor[i] = 0
            else:
                Shadowfactor[i] = 1
        print(Shadowfactor[i])
    """in all other cases there is no point in the same direction as the sun that is higher
    so the shadowfactor is 1: the point receives radiation"""
    return Shadowfactor


def geometricProperties(data,gridboxsize):
    """
    Function that determines the average height over width of an area,
    the average height over width, and the built fraction of an area
    :param data: height data of city
    :return:
    H_W : height over width ratio
    ave_height : average height of the area
    delta: fraction of built area
    """
    ave_height = np.mean(data[data>0])
    """The road elements are actually also water elements"""
    road_elements = np.count_nonzero(data==0)
    built_elements = np.count_nonzero(data>0)
    water_elements = np.count_nonzero(data==-1)
    delta = built_elements/(road_elements+built_elements+water_elements)
    """We want to determine the wall area from the height and delta
    Say each block is a separate building: then the wall area would be 4*sum(builtarea), 
    but since we have a certain density of houses we could make a relation 
    between density and buildings next to each other"""
    Roof_area = built_elements*gridboxsize**2
    Road_area = road_elements*gridboxsize**2
    Water_area = water_elements*gridboxsize**2
    [Wall_area,wall_area_total] = wallArea(data)

    Total_area = Roof_area + wall_area_total + Road_area + Water_area
    """Fractions of the area of the total surface"""
    Roof_frac = np.around(Roof_area/Total_area,3)
    Wall_frac = np.around(wall_area_total/Total_area,3)
    Road_frac = np.around(Road_area/Total_area,3)
    Water_frac = np.around(Water_area/Total_area,3)
    return ave_height, delta, Roof_area, wall_area_total, Road_area,Water_area, Roof_frac, Wall_frac, Road_frac, Water_frac

def wallArea(data):
    """Matrix of ones where there are buildings"""
    [x_len,y_len] = [data.shape[0],data.shape[1]]
    """Set all the water elements to 0 height again"""
    data[data<0] = 0
    """We only evaluate the area in the center block"""
    wall_area = np.ndarray([int(x_len/2),int(y_len/2)])
    for i in range(int(x_len/4),int(3*x_len/4)):
        for j in range(int(y_len/4),int(3*y_len/4)):
            if (data[i,j]>0):
                """We check for all the points surrounding the building if they are also buildings, 
                if the building next to it is higher the wall belongs to the building next to it,
                if the current building is higher, the exterior wall is the difference in height * gridboxsize"""
                wall1 = max(data[i,j]-data[i+1,j],0)*gridboxsize
                wall2 = max(data[i,j]-data[i-1,j],0)*gridboxsize
                wall3 = max(data[i,j]-data[i,j+1],0)*gridboxsize
                wall4 = max(data[i,j]-data[i,j-1],0)*gridboxsize
                """The wall area corresponding to that building is"""
                wall_area[int(i-x_len/4),int(j-y_len/4)] = wall1+wall2+wall3+wall4
            elif (data[i,j]==0):
                wall_area[int(i-x_len/4),int(j-y_len/4)] = 0
    """wall_area is a matrix of the size of center block of data, 
    with each point storing the the exterior wall for that building,
    wall_area_total is the total exterior wall area of the dataset"""
    wall_area_total = np.sum(wall_area)
    return wall_area, wall_area_total

data = datasquare(dtm1,dsm1,dtm2,dsm2,dtm3,dsm3,dtm4,dsm4)
coords = coordheight(data)
blocklength = int((data.shape[0]/2*data.shape[1]/2))
#calc_SVF(coords, steps_psi , steps_beta,max_radius,blocklength)
# [ave_height, delta, Roof_area, Wall_area, Road_area,Water_area, Roof_frac, Wall_frac, Road_frac, Water_frac] = geometricProperties(data,gridboxsize)
# print(ave_height, delta, Roof_area, Wall_area, Road_area,Water_area, Roof_frac, Wall_frac, Road_frac, Water_frac)
#

# plt.figure()
# datasq = readdata(minheight,dsm1,dtm1)
# datasq[datasq<0] = 0
# plt.imshow(datasq, cmap=plt.get_cmap('gray'))
# plt.show()
shadowfactor(coords, 286,Constants.latitude,Constants.long_rd,10.5,steps_beta,blocklength)

