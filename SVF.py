import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import multiprocessing
import tifffile as tf
from tqdm import tqdm
import config
from functools import partial
import time

import KNMI_SVF_verification
import Constants
import Sunpos

sttime = time.time()

input_dir = config.input_dir
"""Now we want to calculate the sky view factor"""
steps_beta = 360 # so we range in steps of 2 degrees
steps_psi = 90 # so we range in steps of 2 degrees
max_radius = 500 # max radius is 500 m
"""define the gridboxsize of the model"""
gridboxsize = 5
gridboxsize_knmi = 0.5
"""objects below 1 m we do not look at"""
minheight = 1

"""DSM's and DTM's"""
"""we need 4 databoxes to account for the SVF's on the side"""
# # linksboven
# dtm1 = "".join([input_dir, '/M5_37EZ2.TIF'])
# dsm1 = "".join([input_dir, '/R5_37EZ2.TIF'])
# # rechtsboven
# dtm2 = "".join([input_dir, '/M5_37FZ1.TIF'])
# dsm2 = "".join([input_dir, '/R5_37FZ1.TIF'])
# # linksonder
# dtm3 = "".join([input_dir, '/M5_37GN2.TIF'])
# dsm3 = "".join([input_dir, '/R5_37GN2.TIF'])
# # rechtsonder
# dtm4 = "".join([input_dir, '/M5_37HN1.TIF'])
# dsm4 = "".join([input_dir, '/R5_37HN1.TIF'])

# linksboven
dtm1 = "".join([input_dir, '/M5_37HN1.TIF'])
dsm1 = "".join([input_dir, '/R5_37HN1.TIF'])
# rechtsboven
dtm2 = "".join([input_dir, '/M5_37HN2.TIF'])
dsm2 = "".join([input_dir, '/R5_37HN2.TIF'])
# linksonder
dtm3 = "".join([input_dir, '/M5_37HZ1.TIF'])
dsm3 = "".join([input_dir, '/R5_37HZ1.TIF'])
# rechtsonder
dtm4 = "".join([input_dir, '/M5_37HZ2.TIF'])
dsm4 = "".join([input_dir, '/R5_37HZ2.TIF'])


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
    # x_len = x_len-2*max_radius
    # y_len = y_len-2*max_radius
    coords = np.ndarray([x_len*y_len,3])
    """ so we start with the list of coordinates with all the points we want to evaluate
    all other points are after that, for this we use 2 different counters."""
    rowcount_block = (x_len-2*max_radius)*(y_len-2*max_radius) #int((x_len/2)*(y_len/2))
    rowcount_center = 0
    """we need to make a list of coordinates where the center block is first"""
    for i in range(x_len):
        for j in range(y_len):
            #if ((x_len/4)<=i and i<(3*x_len/4) and (y_len/4)<=j and j<(3*y_len/4)):
            if ((max_radius)<=i and i<(x_len-max_radius) and (max_radius)<=j and j<(y_len-max_radius)):
                coords[rowcount_center,0] = i
                coords[rowcount_center,1] = j
                coords[rowcount_center,2] = data[i,j]
                rowcount_center += 1
            #elif (i<(x_len/4) or i>=(3*x_len/4) or j<(y_len/4) or j>=(3*y_len/4)):
            elif (i<(max_radius) or i>=(x_len-max_radius) or j<(max_radius) or j>=(y_len-max_radius)):
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
    dx = (coord[:,1]-point[1])*gridboxsize
    # Rows is dy
    dy = (coord[:,0]-point[0])*gridboxsize
    dist = np.sqrt(abs(dx)**2 + abs(dy)**2)
    """angle is 0 north direction"""
    angle = np.arctan2(dy,dx)+np.pi/2
    return dist,angle

def dome(point, coords, maxR):
    """
    :param point: point we are evaluating
    :param coords: array of coordinates with heights
    :param maxR: maximum radius in which we think the coordinates can influence the SVF
    :return: a dome of points that we take into account to evaluate the SVF
    """

    radii, angles = dist(point,coords)
    coords = np.column_stack([coords, radii])
    coords = np.column_stack([coords, angles])
    """the dome consist of points higher than the view height and within the radius we want"""
    dome = coords[(np.logical_and(coords[:,3]<maxR,coords[:,3]>0.1)),:]
    dome = dome[(dome[:,2]>point[2]),:]
    return dome

def d_area(psi,steps_beta,maxR):

    """Radius at ground surface and at the height of the projection of the building"""
    d_area = 2*np.pi/steps_beta*maxR**2*np.sin(psi)
    return d_area

def SkyViewFactor(point, coords, max_radius):
    betas_lin = np.linspace(0,2*np.pi,steps_beta)
    """this is the analytical dome area but should make same assumption as for d_area"""
    dome_area = max_radius**2*2*np.pi
    """ we throw away all point outside the dome
    # dome is now a 5 column array of points:
    # the 5 columns: x,y,z,radius,angle theta"""
    dome_p = dome(point, coords, max_radius)
    betas = np.zeros(steps_beta)

    """we loop over all points in the dome"""
    d = 0
    while (d < dome_p.shape[0]):

        psi = np.arctan((dome_p[d,2]-point[2])/dome_p[d,3])
        """The angles of the min and max angle of the building"""
        beta_min = - np.arcsin(np.sqrt(2*gridboxsize**2)/2/dome_p[d,3]) + dome_p[d,4]
        beta_max = np.arcsin(np.sqrt(2*gridboxsize**2)/2/dome_p[d,3]) + dome_p[d,4]

        """Where the index of betas fall within the min and max beta, and there is not already a larger psi blocking"""
        betas[np.nonzero(np.logical_and((betas < psi), np.logical_and((beta_min <= betas_lin), (betas_lin < beta_max))))] = psi
        d += 1
    areas = d_area(betas, steps_beta, max_radius)
    """The SVF is the fraction of area of the dome that is not blocked"""
    SVF = np.around((dome_area - np.sum(areas))/dome_area, 3)
    print(SVF)
    return SVF

def calc_SVF(coords, max_radius, blocklength):
    """
    Function to calculate the sky view factor.
    We create a dome around a point with a certain radius,
    and for each point in the dome we evaluate of the height of this point blocks the view
    :param coords: all coordinates of our dataset
    :param max_radius: maximum radius we think influences the svf
    :param blocklength: the first amount of points in our data set we want to evaluate
    :return: SVF for all points
    """


    def parallel_runs_SVF():
        points = [coords[i,:] for i in range(blocklength)]
        pool = Pool()
        SVF_list = []
        SVF_par = partial(SkyViewFactor, coords=coords,max_radius=max_radius) # prod_x has only one argument x (y is fixed to 10)
        SVF = pool.map(SVF_par, points)
        pool.close()
        pool.join()
        if SVF is not None:
            SVF_list.append(SVF)
        return SVF_list

    if __name__ == '__main__':
        return parallel_runs_SVF()

def calc_SF(coords,Julianday,latitude,longitude,LMT,blocklength):
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
    def parallel_runs_SF():
        points = [coords[i,:] for i in range(blocklength)]
        SF_list = []
        pool = Pool()
        SF_par = partial(shadowfactor, coords=coords, julianday=Julianday,latitude=latitude,longitude=longitude,LMT=LMT,blocklength=blocklength) # prod_x has only one argument x (y is fixed to 10)
        SF = pool.map(SF_par, points)
        pool.close()
        pool.join()
        if SF is not None:
            SF_list.append(SF)
        return SF_list

    if __name__ == '__main__':
        return parallel_runs_SF()


def shadowfactor(point, coords, julianday,latitude,longitude,LMT,blocklength):
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
    radii, angles = dist(point,coords)
    [azimuth,elevation_angle] = Sunpos.solarpos(julianday,latitude,longitude,LMT)
    beta_min = np.asarray(- np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)
    beta_max = np.asarray(np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)
    if np.count_nonzero(coords[np.logical_and((np.logical_and((angles > beta_min), (angles < beta_max))), ((np.tan(elevation_angle)*radii)<(coords[:,2]-point[2]))),:])>0:
        Shadowfactor = 0
    else:
        Shadowfactor = 1
    """in all other cases there is no point in the same direction as the sun that is higher
    so the shadowfactor is 1: the point receives radiation"""
    return Shadowfactor

def reshape_SVF(data,coords,julianday,lat,long,LMT,reshape,save_CSV,save_Im):
    #
    [x_len, y_len] = [int(data.shape[0]/2),int(data.shape[1]/2)]
    blocklength = int(x_len*y_len)
    "Compute SVF and SF and Reshape the shadow factors and SVF back to nd array"
    SVFs = calc_SVF(coords,max_radius,blocklength)
    SFs = calc_SF(coords,julianday,lat,long,LMT,blocklength)
    "If reshape is true we reshape the arrays to the original data matrix"
    if reshape == True:
        SVF_matrix = np.ndarray([x_len,y_len])
        SF_matrix = np.ndarray([x_len,y_len])
        for i in range(blocklength):
            SVF_matrix[coords[i,0],coords[i,1]]  = SVFs[i]
            SF_matrix[coords[i,0],coords[i,1]] = SFs[i]
        if save_CSV == True:
            np.savetxt("SVFmatrix.csv", SVF_matrix, delimiter=",")
            np.savetxt("SFmatrix.csv", SF_matrix, delimiter=",")
        if save_Im == True:
            tf.imwrite('SVF_matrix.tif', SVF_matrix, photometric='minisblack')
            tf.imwrite('SF_matrix.tif', SF_matrix, photometric='minisblack')
        return SF_matrix,SF_matrix

    elif reshape == False:
        if save_CSV == True:
            np.savetxt("SVFs.csv", SVFs, delimiter=",")
            np.savetxt("SFs.csv", SFs, delimiter=",")
        return SVFs, SFs



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
    [Wall_area, wall_area_total] = wallArea(data)

    Total_area = Roof_area + wall_area_total + Road_area + Water_area
    """Fractions of the area of the total surface"""
    Roof_frac = np.around(Roof_area/Total_area,3)
    Wall_frac = np.around(wall_area_total/Total_area,3)
    Road_frac = np.around(Road_area/Total_area,3)
    Water_frac = np.around(Water_area/Total_area,3)

    H_W = ave_height*delta
    return ave_height, delta, Roof_area, wall_area_total, Road_area,Water_area, Roof_frac, Wall_frac, Road_frac, Water_frac, H_W


def wallArea(data):
    """Matrix of ones where there are buildings"""
    [x_len,y_len] = [data.shape[0], data.shape[1]]
    """Set all the water elements to 0 height again"""
    data[data<0] = 0
    """We only evaluate the area in the center block"""
    wall_area = np.ndarray([int(x_len/2),int(y_len/2)])
    i = int(x_len/4)
    j = int(y_len/4)
    while i < int(3*x_len/4):
        while j < int(3*y_len/4):
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
            i+=1
            j+=1
    """wall_area is a matrix of the size of center block of data, 
    with each point storing the the exterior wall for that building,
    wall_area_total is the total exterior wall area of the dataset"""
    wall_area_total = np.sum(wall_area)
    return wall_area, wall_area_total

# datasq = datasquare(dtm1,dsm1,dtm2,dsm2,dtm3,dsm3,dtm4,dsm4)
# coords = coordheight(datasq)
#blocklength = int(datasq.shape[0]/2*datasq.shape[1]/2)

# [SVF_matrix, SF_matrix] = reshape_SVF(datasq,coords,Constants.julianday,Constants.latitude,Constants.long_rd,Constants.hour,reshape=False,save_CSV=False,save_Im=False)
# print(SVF_matrix)
# endtime = time.time()


"""What if we iterate over the middle block of 1 block instead?"""
data = readdata(minheight,dsm1,dtm1)
coords = coordheight(data)
blocklength = int(data.shape[0]-2*max_radius)*(data.shape[1]-2*max_radius) #int((x_len/2)*(y_len/2))
[SVF_matrix,SF_matrix] = reshape_SVF(data,coords,Constants.julianday,Constants.latitude,Constants.long_rd,Constants.hour,reshape=False,save_CSV=False,save_Im=False)
print(SVF_matrix)
print(SF_matrix)
# print(multiprocessing.cpu_count())
#print(KNMI_SVF_verification.Verification(SVF_matrix,KNMI_SVF_verification.SVF_knmi1,gridboxsize,gridboxsize_knmi))
endtime = time.time()
elapsed_time = endtime-sttime
print('Execution time:', elapsed_time, 'seconds')
