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
steps_theta = 180 # so we range in steps of 2 degrees
steps_phi = 90 # so we range in steps of 2 degrees
max_radius = 100 # max radius is 100 m
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
    data[data > 10 ** 38] = 0  # remove extreme large numbers for water and set to zero.
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
    datadiffcopy[data_diff < minheight] = 0
    data_final = datadiffcopy
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
    dx = (point[1] - coord[1])*gridboxsize
    # Rows is dy
    dy = (point[0] - coord[0])*gridboxsize
    dist = np.sqrt(abs(dx)**2 + abs(dy)**2)
    """Define the angle the same way we define the azimuth: 0 for """
    angle = np.arctan(dy/dx)
    return dist,angle

def dist_round(point, coord,steps_theta):
    """
    :param point: evaluation point (x,y,z)
    :param coord: array of coordinates with heights
    :return: the distance from each coordinate to the point and the angle
    """
    dx = (coord[1] - point[1])*gridboxsize
    dy = (coord[0] - point[0])*gridboxsize
    dist = np.sqrt(abs(dx)**2 + abs(dy)**2)
    angle = np.arctan(dy/dx)
    #thetas = np.linspace(0,2*np.pi,steps_theta+1)
    #angle_rounded = thetas[(abs(thetas-angle)).argmin()]
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
        radius[i],angle[i] = dist_round(point,coords[i],steps_theta)
    coords = np.concatenate([coords,radius],axis=1)
    coords = np.concatenate([coords,angle],axis=1)
    """the dome consist of points higher than the view height and within the radius we want"""
    dome = coords[(np.logical_and(coords[:,3]<maxR,coords[:,3]>0.1)),:]
    dome = dome[(dome[:,2]>point[2]),:]
    return dome

def d_area(radius, d_theta,height,maxR):
    """we assume the dome is large, and d_theta and d_phi small enough,
     such that we can approximate the d_area as a flat surface"""
    dx = radius*2*np.sin(d_theta/2)
    """multiply by maxR over radius such that the small area gets projected on the dome"""
    d_area = (dx*height)*maxR/radius
    return d_area

def calc_SVF(coords, steps_phi , steps_theta,max_radius,blocklength):
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
    phi_lin = np.linspace(0,np.pi,steps_phi)
    thetas = np.zeros(steps_theta)
    d_theta = 2*np.pi/steps_theta
    d_phi = np.pi/steps_phi
    thetas_lin = np.linspace(0,2*np.pi,steps_theta+1)
    SVF = np.ndarray([blocklength,1])

    for i in tqdm(range(blocklength),desc="loop over points"):
        point = coords[i,:]
        """ we throw away all point outside the dome
        # dome is now a 5 column array of points:
        # the 5 columns: x,y,z,radius,angle theta"""
        dome_p = dome(point, coords, max_radius)
        """we loop over all points inside the dome"""
        for d in tqdm(range(dome_p.shape[0]),desc="dome loop"):
            for p in range(phi_lin.shape[0]):
                """evaluate if the height of the point is higher than the height of the point"""
                h_phi = np.tan(phi_lin[p])*(dome_p[d,3]) #
                heightdif = dome_p[d,2]-point[2]
                if (heightdif>=h_phi):
                    """ if there already exist a blocking for that angle theta:
                    calculate of the area of this blocking is more"""
                    area = d_area(dome_p[d,3],d_theta,heightdif,max_radius)
                    angle_rounded = thetas[(abs(thetas-dome_p[d,4])).argmin()]
                    if (thetas[angle_rounded]<area):
                        thetas[angle_rounded] = area
        """this is the analytical dome area but should make same assumption as for d_area
        dome_area = max_radius**2*2*np.pi"""
        dome_area = (max_radius*np.sin(d_phi/2)*2)*(max_radius*np.sin(d_theta/2)*2)*steps_theta*steps_phi
        """The SVF is the fraction of area of the dome that is not blocked"""
        SVF[i] = (dome_area - np.sum(thetas,0))/dome_area
    return SVF

def calc_SVF_try2(coords, steps_phi , steps_theta,max_radius,blocklength):
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
    thetas = np.linspace(0,2*np.pi,steps_phi+1)
    theta_array = np.zeros(len(thetas))
    d_theta = 2*np.pi/steps_theta
    d_phi = np.pi/steps_phi

    SVF = np.ndarray([blocklength,1])

    for i in tqdm(range(blocklength),desc="loop over points"):
        point = coords[i,:]
        """ we throw away all point outside the dome
        # dome is now a 5 column array of points:
        # the 5 columns: x,y,z,radius,angle theta"""
        dome_p = dome(point, coords, max_radius)
        """we loop over theta"""
        for d in tqdm(range(dome_p.shape[0]),desc="dome loop"):
            for t in range(steps_theta):
                if (dome_p[d,4]==thetas[t]):
                    heightdif = dome_p[d,2]-point[2]
                    area = d_area(dome_p[d,3],d_theta,heightdif,max_radius)
                    theta_array[t] = max(theta_array[t], area)

        """this is the analytical dome area but should make same assumption as for d_area
        dome_area = max_radius**2*2*np.pi"""
        dome_area = (max_radius*np.sin(d_phi/2)*2)*(max_radius*np.sin(d_theta/2)*2)*steps_theta*steps_phi
        """The SVF is the fraction of area of the dome that is not blocked"""
        SVF[i] = (dome_area - np.sum(thetas,0))/dome_area
    return SVF

def shadowfactor(coords, julianday,latitude,longitude,LMT,steps_theta,blocklength):
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
    #block_lin = np.arange(0,blocklength,1000)
    d_theta = 2*np.pi/steps_theta
    thetas = np.linspace(0,2*np.pi,steps_theta+1)
    #azi_round = thetas[(abs(thetas-azimuth)).argmin()]
    for i in tqdm(range(blocklength),desc="loop over points in block for Shadowfactor"):
        """the point we are currently evaluating"""
        Shadowfactor[i] = 1
        point = coords[i,:]

        for j in range(coords.shape[0]):
            radius, angle = dist(point,coords[j])
            """The angle we measure is the measure from east, and positive if we turn upwards, 
            the azimuth is defined as 0 for when the sun is south, 
            and positive westwards, to compare with the azimuth angle we thus have to correct the angle with pi/2"""
            angle = (angle+np.pi/2)
            """if the angle is within a very small range as the angle of the sun"""
            if (angle <= (azimuth+d_theta) and angle >= azimuth-d_theta):
                """if the elevation angle times the radius is smaller than the height of that point
                the shadowfactor is zero since that point blocks the sun"""
                if ((np.tan(elevation_angle)*radius)<(coords[j,2]-point[2])):
                    Shadowfactor[i]=0
        print(Shadowfactor[i])
    """in all other cases there is no point in the same direction as the sun that is higher
    so the shadowfactor is 1: the point receives radiation"""
    return Shadowfactor


def height_width(data,gridboxsize):
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
    delta = built_elements/(road_elements+built_elements)
    """We want to determine the wall area from the height and delta
    Say each block is a separate building: then the wall area would be 4*sum(builtarea), 
    but since we have a certain density of houses we could make a relation 
    between density and buildings next to each other"""
    Roof_area = built_elements*gridboxsize**2
    Road_area = road_elements*gridboxsize**2
    """As a first approximation we assume that if the building density is delta,
    The wall area decreases with factor (1-delta); i.e. (1-delta) of the walls faces another wall"""
    Wall_area = 4*np.sum(data)*gridboxsize*(1-delta)

    Total_area = Roof_area+Wall_area+Road_area
    """Fractions of the area of the total surface"""
    Roof_frac = np.around(Roof_area/Total_area,3)
    Wall_frac = np.around(Wall_area/Total_area,3)
    Road_frac = np.around(Road_area/Total_area,3)
    return ave_height, delta, Roof_area, Wall_area, Road_area, Roof_frac, Wall_frac, Road_frac

def wallArea(data):
    """Matrix of ones where there are buildings"""
    [x_len,y_len] = [data.shape[0],data.shape[1]]

    wall_area = np.ndarray([x_len,y_len])
    for i in range(int(x_len/2),int(x_len/2+x_len-1)):
        for j in range(int(y_len/2),int(y_len/2+y_len-1)):
            if (data[i,j]>0):
                wall1 = max(data[i,j]-data[i+1,j],0)*gridboxsize
                wall2 = max(data[i,j]-data[i-1,j],0)*gridboxsize
                wall3 = max(data[i,j]-data[i,j+1],0)*gridboxsize
                wall4 = max(data[i,j]-data[i,j-1],0)*gridboxsize
                """The wall area corresponding to that building is"""
                wall_area[i,j] = wall1+wall2+wall3+wall4
            elif (data[i,j]==0):
                wall_area[i,j] = 0
    return wall_area

data = datasquare(dtm1,dsm1,dtm2,dsm2,dtm3,dsm3,dtm4,dsm4)
#coords = coordheight(data)
[ave_height, delta, Roof_area, Wall_area, Road_area, Roof_frac, Wall_frac, Road_frac] = height_width(data,gridboxsize)
print(ave_height, delta, Roof_area, Wall_area, Road_area, Roof_frac, Wall_frac, Road_frac)
blocklength = int((data.shape[0]/2*data.shape[1]/2))
wall_area = wallArea(data)

print(wall_area)


