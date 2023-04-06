import numpy as np
#import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import tifffile as tf
from tqdm import tqdm
import config
from functools import partial
import time
#import KNMI_SVF_verification
import Constants
import Sunpos
# import SVFs05m
# import SF05mHN1
# import pickle
# import SVFs5m
# import SVFGR25
# from pynverse import inversefunc
# from scipy.optimize import curve_fit
# from mpl_toolkits.mplot3d import Axes3D


sttime = time.time()

input_dir = config.input_dir
"""Now we want to calculate the sky view factor"""
steps_beta = 360 # so we range in steps of 1 degree
"""define the gridboxsize of the model"""
gridboxsize = 0.5
if gridboxsize==5:
    max_radius = 500
elif gridboxsize==0.5:
    max_radius = 100
gridboxsize_knmi = 0.5
"objects below 1 m we do not look at"
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

# # linksboven
# dtm1 = "".join([input_dir, '/M5_37HN1.TIF'])
# dsm1 = "".join([input_dir, '/R5_37HN1.TIF'])
# # rechtsboven
# dtm2 = "".join([input_dir, '/M5_37HN2.TIF'])
# dsm2 = "".join([input_dir, '/R5_37HN2.TIF'])
# # linksonder
# dtm3 = "".join([input_dir, '/M5_37HZ1.TIF'])
# dsm3 = "".join([input_dir, '/R5_37HZ1.TIF'])
# # rechtsonder
# dtm4 = "".join([input_dir, '/M5_37HZ2.TIF'])
# dsm4 = "".join([input_dir, '/R5_37HZ2.TIF'])

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
    [x_len, y_len] = np.shape(data)
    # if (x_len > 1250):
    #     for i in range(x_len):  # maybe inefficient??
    #         for j in range(y_len):
    #             if data_diff[i,j] != 0 and i < (x_len -1) and j < (y_len-1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j+1] ==0 and data_diff[i+1,j-1] ==0 and data_diff[i-1,j+1] == 0 and data_diff[i-1,j-1] ==0:
    #                 data_diff[i, j] = 0
    #             elif data_diff[i,j] != 0 and i == 0 and j < (y_len-1) and data_diff[i+1,j] == 0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j+1] ==0 and data_diff[i+1,j-1] ==0:
    #                 data_diff[i,j] = 0
    #             elif data_diff[i,j] != 0 and i == x_len-1 and j < (y_len-1) and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i-1,j+1] == 0 and data_diff[i-1,j-1] ==0:
    #                 data_diff[i, j] = 0
    #             elif data_diff[i,j] != 0 and j == 0 and i < (x_len-1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i+1,j+1] ==0 and data_diff[i-1,j+1] == 0:
    #                 data_diff[i, j] = 0
    #             elif data_diff[i,j] != 0 and j == (y_len-1) and i < (x_len -1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j-1] ==0 and data_diff[i-1,j-1] ==0:
    #                 data_diff[i, j] = 0

    """filter all heights below the min height out"""
    data_diff[data_diff<minheight] = 0
    """All water elements are set to zero"""
    return data_diff, data_water

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

def MediateData(data,data_water,delta_x,delta_y,delta_z,gridboxsize):
    "In this averaging, the gridcell must be half full to be full or it will be empty"
    "Define what the ratios of gridcells are in x and y direction"
    min_vol = delta_x*delta_y*delta_z
    [x_len,y_len] = data.shape
    GR_x = delta_x/gridboxsize
    GR_y = delta_y/gridboxsize
    if np.logical_or((x_len%GR_x != 0),(y_len%GR_y != 0)):
        print('The chosen gridboxsizes are not appropriate')
    data_new = np.zeros([int(x_len/GR_x),int(y_len/GR_y)])
    data_water_new = np.zeros([int(x_len/GR_x),int(y_len/GR_y)])
    [x_len,y_len] = data_new.shape
    for i in range(x_len):
        for j in range(y_len):
            part = data[int(i*GR_x):int((i+1)*GR_x),int(j*GR_y):int((j+1)*GR_y)]
            [p_x,p_y] = part.shape
            part_water = data_water[int(i*GR_x):int((i+1)*GR_x),int(j*GR_y):int((j+1)*GR_y)]
            Vol = np.mean(part)*delta_x*delta_y
            data_water_new[i,j] = np.round(np.count_nonzero(part_water)/(p_x*p_y))
            data_new[i,j] = np.round(Vol/min_vol)*delta_z
    data_new[data_water_new > 0] = 0
    return data_new,data_water_new

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
    rowcount_center = 0
    "The middle block is evaluated"
    rowcount_block = int(x_len/2*y_len/2)
    for i in range(x_len):
        for j in range(y_len):
            """we need to make a list of coordinates where the center block is first"""
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

def dist(point, coord,gridboxsize):
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
    angle = (np.pi/2 - np.arctan2(dy,dx))
    return dist,angle

def dome(point, coords, maxR,gridboxsize):
    """
    :param point: point we are evaluating
    :param coords: array of coordinates with heights
    :param maxR: maximum radius in which we think the coordinates can influence the SVF
    :return: a dome of points that we take into account to evaluate the SVF
    """

    radii, angles = dist(point,coords,gridboxsize)
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

def SkyViewFactor(point, coords, max_radius,gridboxsize,steps_beta):
    betas_lin = np.linspace(-np.pi/2,3*np.pi/2,steps_beta, endpoint=False)
    "we throw away all point outside the dome"
    # dome is now a 5 column array of points:
    # the 5 columns: x,y,z,radius,angle theta"""
    dome_p = dome(point, coords, max_radius,gridboxsize)
    betas = np.zeros(steps_beta)
    """we loop over all points in the dome"""
    d = 0
    while (d < dome_p.shape[0]):

        psi = np.arctan((dome_p[d,2]-point[2])/dome_p[d,3])
        """The angles of the min and max angle of the building"""
        beta_min = - np.arcsin(gridboxsize/2/dome_p[d,3]) + dome_p[d,4]
        beta_max = np.arcsin(gridboxsize/2/dome_p[d,3]) + dome_p[d,4]

        """Where the index of betas fall within the min and max beta, and there is not already a larger psi blocking"""
        betas[np.nonzero(np.logical_and((betas < psi), np.logical_and((beta_min <= betas_lin), (betas_lin < beta_max))))] = psi
        d += 1

    """The SVF is the fraction of area of the dome that is not blocked"""
    SVF = np.around((np.sum(np.cos(betas)**2)/steps_beta),5)
    return SVF

def calc_SVF(coords, max_radius, blocklength, gridboxsize):
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
        SVF_par = partial(SkyViewFactor, coords=coords,max_radius=max_radius,gridboxsize=gridboxsize) # prod_x has only one argument x (y is fixed to 10)
        SVF = pool.map(SVF_par, points)
        pool.close()
        pool.join()
        if SVF is not None:
            SVF_list.append(SVF)
        return SVF_list

    if __name__ == '__main__':
        result = parallel_runs_SVF()
        return result

def calc_SF(coords,azimuth,elevation_angle,blocklength):
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
        SF_par = partial(shadowfactor, coords=coords, azimuth=azimuth,elevation_angle=elevation_angle) # prod_x has only one argument x (y is fixed to 10)
        SF = pool.map(SF_par, points)
        pool.close()
        pool.join()
        if SF is not None:
            SF_list.append(SF)
        return SF_list

    if __name__ == '__main__':
        return parallel_runs_SF()


def shadowfactor(point, coords, azimuth,elevation_angle):
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
    radii, angles = dist(point,coords,gridboxsize)
    angles = (angles + 2*np.pi) % (2*np.pi)
    beta_min = np.asarray(- np.arcsin(gridboxsize/2/radii) + azimuth)
    beta_max = np.asarray(np.arcsin(gridboxsize/2/radii) + azimuth)
    if np.logical_or(elevation_angle<=0,(np.count_nonzero(coords[np.logical_and((np.logical_and((angles > beta_min), (angles < beta_max))), ((np.tan(elevation_angle)*radii)<(coords[:,2]-point[2]))),:])>0)):
        Shadowfactor = 0
    else:
        Shadowfactor = 1
    """in all other cases there is no point in the same direction as the sun that is higher
    so the shadowfactor is 1: the point receives radiation"""
    return Shadowfactor

def reshape_SVF(data, coords,gridboxsize,azimuth,zenith,reshape,save_CSV,save_Im):
    """
    :param data: AHN data
    :param coords: list of coordinates
    :param gridboxsize: size of gridcell
    :param azimuth: solar azimuth angle
    :param zenith: solar elevation angle
    :param reshape: Boolean: whether to turn the SVF/SF back to matrix form
    :param save_CSV: Boolean: save CSV file of SVF/SF
    :param save_Im: Boolean: save image of SVF/SF
    :return:
    """
    [x_len, y_len] = data.shape
    blocklength = int(x_len/2*y_len/2)
    "Compute SVF and SF and Reshape the shadow factors and SVF back to nd array"
    #SVFs = calc_SVF(coords, max_radius, blocklength, gridboxsize)
    #print(SVFs)
    SFs = calc_SF(coords,azimuth,zenith,blocklength)
    "If reshape is true we reshape the arrays to the original data matrix"
    if (reshape == True) & (SFs is not None):
        #SVF_matrix = np.ndarray([x_len,y_len])
        SF_matrix = np.ndarray([x_len,y_len])
        for i in range(blocklength):
            #SVF_matrix[int(coords[i,0]),int(coords[i,1])] = SVFs[i]
            SF_matrix[int(coords[i,0]),int(coords[i,1])] = SFs[i]
        #SVF_matrix = SVF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
        SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
        if save_CSV == True:
            #np.savetxt("SVFmatrix.csv", SVF_matrix, delimiter=",")
            np.savetxt("SFmatrix.csv", SF_matrix, delimiter=",")
        if save_Im == True:
            #tf.imwrite('SVF_matrix.tif', SVF_matrix, photometric='minisblack')
            tf.imwrite('SF_matrix.tif', SF_matrix, photometric='minisblack')
        return SF_matrix #,SVF_matrix
    #
    elif (reshape == False) & (SFs is not None):
        if save_CSV == True:
            #np.savetxt("SVFs" + str(gridboxsize) + ".csv", SVFs, delimiter=",")
            np.savetxt("SFs" + str(gridboxsize) + ".csv", SFs, delimiter=",")
        return SFs #,SVFs#,


def geometricProperties(data,data_water,grid_ratio,gridboxsize):
    """
    Function that determines the average height over width of an area,
    the average height over width, and the built fraction of an area
    :param data: height data of city
    :return:
    H_W : height over width ratio
    ave_height : average height of the area
    delta: fraction of built area
    """
    data[data<0] = 0
    [x_long, y_long] = data.shape
    # x_long = int(x_long/2)
    # y_long = int(y_long/2)
    [Wall_area, wall_area_total] = wallArea(data,gridboxsize)
    Wall_area_gridcell = np.sum(Wall_area,axis=2)
    delta = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    ave_height = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    road_elements = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    built_elements = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    ground_elements = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    water_elements = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    wall_area_med = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(int(x_long/grid_ratio)):
        for j in range(int(y_long/grid_ratio)):
            part = data[i*grid_ratio:(i+1)*grid_ratio, j*grid_ratio:(j+1)*grid_ratio]
            part_water = data_water[i*grid_ratio:(i+1)*grid_ratio, j*grid_ratio:(j+1)*grid_ratio]
            part_wall = Wall_area_gridcell[i*grid_ratio:(i+1)*grid_ratio, j*grid_ratio:(j+1)*grid_ratio]
            ave_height[i,j] = np.mean(part[part>0])
            built_elements[i,j] = np.count_nonzero(part>0)
            ground_elements[i,j] = np.count_nonzero(part==0)
            "The road elements are actually also water elements"
            water_elements[i,j] = np.count_nonzero(part_water)
            road_elements[i,j] = ground_elements[i,j]-water_elements[i,j]
            delta[i,j] = built_elements[i,j]/(built_elements[i,j]+ground_elements[i,j])
            wall_area_med[i,j] = np.sum(part_wall)
    """We want to determine the wall area from the height and delta
    Say each block is a separate building: then the wall area would be 4*sum(builtarea), 
    but since we have a certain density of houses we could make a relation 
    between density and buildings next to each other"""
    Roof_area = built_elements*gridboxsize**2
    Road_area = road_elements*gridboxsize**2
    Water_area = water_elements*gridboxsize**2
    Ground_area = ground_elements*gridboxsize**2
    Total_area = Roof_area + wall_area_med + Ground_area
    """Fractions of the area of the total surface"""
    Roof_frac = np.around(Roof_area/Total_area,3)
    Wall_frac = np.around(wall_area_med/Total_area,3)
    Road_frac = np.around(Road_area/Total_area,3)
    Water_frac = np.around(Water_area/Total_area,3)
    Ground_frac = np.around(Ground_area/Total_area,3)
    H_W = ave_height * delta
    return Roof_frac, Wall_frac, Road_frac,Water_frac,Ground_frac #Roof_area, wall_area_med, Road_area#

def average_surfacetype(matrix,data, grid_ratio):
    [x_long, y_long] = matrix.shape
    #minheight = 3
    ave_roof = np.ndarray([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    ave_road = np.ndarray([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(int(x_long/grid_ratio)):
        for j in range(int(y_long/grid_ratio)):
            data_part = data[int(i*grid_ratio):int((i+1)*grid_ratio), int(j*grid_ratio):int((j+1)*grid_ratio)]
            part = matrix[int(i*grid_ratio):int((i+1)*grid_ratio), int(j*grid_ratio):int((j+1)*grid_ratio)]
            ave_roof[i,j] = np.mean(part[data_part>0])
            ave_road[i,j] = np.mean(part[data_part<=0])
    ave_roof[np.isnan(ave_roof)] = np.nanmean(ave_roof)
    ave_road[np.isnan(ave_road)] = np.nanmean(ave_road)
    return ave_roof,ave_road

def average_svf(SVF_matrix, grid_ratio):
    [x_long, y_long] = SVF_matrix.shape
    SVF_ave = np.ndarray([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(int(x_long/grid_ratio)):
        for j in range(int(y_long/grid_ratio)):
            part = SVF_matrix[int(i*grid_ratio):int((i+1)*grid_ratio), int(j*grid_ratio):int((j+1)*grid_ratio)]
            SVF_ave[i,j] = np.mean(part)
    return SVF_ave

def SF_wall(point,coords,type,wall_len,azimuth,elevation_angle):
    "Now the shadowfactor is 0 if the top of the wall surface does not receive radiation. "
    " But sometimes (most of the times) part of the wall is blocked from sunlight, how to solve this??"
    if type==0:
        coords = coords[(coords[:,0]<point[0]),:]
    elif type==1:
        coords = coords[(coords[:,1]>point[1]),:]
    elif type==2:
        coords = coords[(coords[:,0]>point[0]),:]
    elif type==3:
        coords = coords[(coords[:,1]<point[1]),:]

    radii, angles = dist(point,coords,gridboxsize)

    if type==3:
        angles = (angles + 2*np.pi) % (2*np.pi)

    beta_min = np.asarray(- np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)
    beta_max = np.asarray(np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)

    num_slices = int(np.rint(wall_len))
    psi = np.ndarray((num_slices,1))
    d_h = wall_len/num_slices
    Shadowfactor = np.ndarray((num_slices,1))

    for p in range(len(psi)):
        h = point[2] - wall_len + p*d_h
        if np.logical_and((np.logical_or((np.count_nonzero(coords[np.logical_and((np.logical_and((angles > beta_min), (angles < beta_max))), ((np.tan(elevation_angle)*radii)<(coords[:,2]-h))),:])>0),type==0)),np.min(angles)<=azimuth<=np.max(angles)):
            Shadowfactor[p] = 0
        else:
            Shadowfactor[p] = 1
    """in all other cases there is no point in the same direction as the sun that is higher
    so the shadowfactor is 1: the point receives radiation"""
    SF = np.mean(Shadowfactor)
    return SF

def wallArea(data,gridboxsize):
    """
    :param data: Dataset to compute the wall area over
    :param gridboxsize: size of the grid cells
    :return: the wallarea matrix and total wall area
    """
    """Matrix of ones where there are buildings"""
    [x_len,y_len] = data.shape
    """Set all the water elements to 0 height again"""
    data[data<0] = 0
    """We only evaluate the area in the center block"""
    "The 3d wall area matrix is the size of data with 4 rows for each wall in order north, east, south west"
    wall_area = np.zeros([int(x_len),int(y_len),4])

    for i in range(int(x_len/4),int(3*x_len/4)):
        for j in range(int(y_len/4),int(3*y_len/4)):
            if (data[i,j]>0):
                """We check for all the points surrounding the building if they are also buildings, 
                if the building next to it is higher the wall belongs to the building next to it,
                if the current building is higher, the exterior wall is the difference in height * gridboxsize"""
                wall_area[i,j,0] = max(data[i,j]-data[i-1,j],0)*gridboxsize
                wall_area[i,j,1] = max(data[i,j]-data[i,j+1],0)*gridboxsize
                wall_area[i,j,2] = max(data[i,j]-data[i+1,j],0)*gridboxsize
                wall_area[i,j,3] = max(data[i,j]-data[i,j-1],0)*gridboxsize
                """The wall area corresponding to that building is"""
                #wall_area[int(i-x_len/4),int(j-y_len/4)] = wall1+wall2+wall3+wall4
            elif (data[i,j]==0):
                wall_area[int(i-x_len/4),int(j-x_len/4),:] = 0
    """wall_area is a matrix of the size of center block of data, 
    with each point storing the exterior wall for that building,
    wall_area_total is the total exterior wall area of the dataset"""
    wall_area = wall_area[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4),:]
    wall_area_total = np.sum(wall_area)
    return wall_area, wall_area_total

def fisheye():
    [x_len,y_len] = data.shape
    coords = coordheight(data,gridboxsize)
    "Make fisheye plot"
    blocklength = x_len/2*y_len/2
    point = coords[int(blocklength),:]
    bottom = 0
    max_area = max_radius**2 * 2 * np.pi / steps_beta
    [svf, betas] = SkyViewFactor(point,coords,max_radius,gridboxsize)
    areas = (np.cos(betas)**2)
    theta = np.linspace(0.0, 2 * np.pi, steps_beta, endpoint=False)
    radii = - areas + max_area
    width = (2*np.pi) / steps_beta

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=bottom)
    ax.set_facecolor("grey")
    ax.get_yaxis().set_ticks([])
    ax.get_yaxis().set_visible(False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor("lightblue")
        bar.set_alpha(0.8)

    plt.show()


def SVF_wall(point,coords,maxR,type,wall_len):
    "For a wall point determine whether it is north, south east or west faced."
    "Retrieve all points inside a max radius"
    "Point is a point on the dataset that is either north, west, east or south facing"
    "wall area is point []"
    point_zero = np.copy(point)
    point_zero[2] = point[2]-wall_len
    dome_zero = dome(point_zero,coords,maxR,gridboxsize)
    "We have a point on data that is elevated: " \
    "for this point we have wallmatrix that has dimensions [data.shape[0],data.shape[1],4]" \
    "So for the SVF for wallmatrix[i,j,0] (north facing)"
    "Use this function in a loop where we loop over all data points in coords > 0 (all buildings), " \
    "then call this function while looping over all 4 sides with w in wall" \
    "SVF_WVF_wall(point,coords,maxR,type=w,wall_area[point[0],point[1],w])"

    num_slices = int(np.rint(wall_len))
    if (type==0): #Northern facing wall
        dome_zero = dome_zero[dome_zero[:,0]<point[0]] #so only northern points
        beta_lin = np.linspace(-np.pi/2,np.pi/2,int(steps_beta/2),endpoint=False)
    elif (type==1): #east facing wall
        dome_zero = dome_zero[dome_zero[:,1]>point[1]] #so only eastern points
        beta_lin = np.linspace(0,np.pi,int(steps_beta/2),endpoint=False)
    elif (type==2): #south facing wall
        dome_zero = dome_zero[dome_zero[:,0]>point[0]] #so only southern points
        beta_lin = np.linspace(np.pi/2,3*np.pi/2,int(steps_beta/2),endpoint=False)
    elif (type==3): #west facing wall
        dome_zero = dome_zero[dome_zero[:,1]<point[1]] #so only western points
        dome_zero[:,4] = (dome_zero[:,4]+(2*np.pi))%(2*np.pi)
        beta_lin = np.linspace(np.pi,2*np.pi,int(steps_beta/2),endpoint=False)
    "Now we have all the points that possibly block the view"
    betas = np.zeros(int(steps_beta/2))
    betas_zero = np.zeros(int(steps_beta/2))
    # store the radii of closest blockings
    closest = np.zeros(int(steps_beta/2))
    psi = np.ndarray((num_slices,1))
    len_d = wall_len/num_slices

    "Loop over beta to find the largest blocking in every direction"
    for d in range(dome_zero.shape[0]):
        "The Angle between the height of the blocking and the height of the point on the wall with the horizontal axis"
        #psi_zero = np.arctan((dome_zero[d,2]-(point[2]-wall_area[2]))/dome_zero[d,3])
        """The angles of the min and max horizontal angle of the building"""
        beta_min = - np.arcsin(gridboxsize/2/dome_zero[d,3]) + dome_zero[d,4]
        beta_max = np.arcsin(gridboxsize/2/dome_zero[d,3]) + dome_zero[d,4]

        for p in range(len(psi)):
            psi[p] = np.arctan((dome_zero[d,2] - (point_zero[2] + p*len_d))/dome_zero[d,3])
        psi_ave = np.max(np.mean(psi),0)
        """Where the index of betas fall within the min and max beta, and there is not already a larger psi blocking"""
        betas[np.nonzero(np.logical_and(betas<psi_ave,(np.logical_and((beta_min <= beta_lin),(beta_lin < beta_max)))))] = psi_ave
        #betas_zero[np.nonzero(np.logical_and(betas_zero<psi_zero,(np.logical_and((beta_min <= beta_lin),(beta_lin < beta_max)))))] = psi_zero
    #print(betas)
        # if dome_zero[d,2]==0:
        #     closest[np.nonzero(np.logical_and((closest > dome_zero[d,3]), np.logical_and((beta_min <= beta_lin), (beta_lin < beta_max))))] = dome_zero[d,3]
    SVF_wall = np.around((np.sum(np.cos(betas)**2)/steps_beta),3)
    return SVF_wall

def Inv_WallvsRoadMasson(SVF_road):
    h_w = (1-SVF_road**2)/(2*SVF_road)
    SVF_wall = (1/2*(h_w+1-np.sqrt(h_w**2+1))/h_w)
    SVF_wall[np.isnan(SVF_wall)] = np.nanmean(SVF_wall)
    return SVF_wall

def SF_masson(h_w,Zenith):
    theta_zero = np.arctan(1/h_w)
    SF_roof = 1
    if (Zenith > theta_zero):
        SF_wall = (1/2/h_w)
        SF_road = 0
    elif (Zenith < theta_zero):
        SF_wall = (1/2*np.tan(Zenith))
        SF_road = 1-h_w*np.tan(Zenith)
    if Zenith>np.pi/2:
        SF_wall = 0
        SF_road = 0
        SF_roof = 0
    return SF_roof, SF_wall, SF_road

def SVF_masson(h_w):
    """
    :param h_w: Height over width ratio
    :return: returns the SVF
    """
    SVF_road = (np.sqrt(h_w**2+1)-h_w)
    SVF_wall = (1/2*(h_w+1-np.sqrt(h_w**2+1))/h_w)
    SVF_roof = 1
    return SVF_roof, SVF_wall, SVF_road
""""""

def SF_ave_masson(h_w,Zenith):
    theta_zero = np.arcsin(min((1/(h_w*np.tan(Zenith))),1))
    SF_roof = 1
    SF_road = 2*theta_zero/np.pi - 2/np.pi*h_w*np.tan(Zenith)*(1-np.cos(theta_zero))
    SF_wall = 1/h_w*(1/2-theta_zero/np.pi)+1/np.pi*np.tan(Zenith)*(1-np.cos(theta_zero))
    return SF_roof, SF_wall, SF_road

"Curve fit"
def Wall_roadSF_fit(x,a,b,c):
    return a*x**b*np.tanh(c*x)

def WallSF_fit(Zenith,SF_road):
    H_w = np.linspace(0.2,5,30)
    SF_w = np.empty((len(H_w)))
    SF_r = np.empty((len(H_w)))
    for h in range(len(H_w)):
        [SF_roof, SF_w[h], SF_r[h]] = SF_ave_masson(H_w[h],Zenith)
    popt, pcov = curve_fit(Wall_roadSF_fit, SF_r, SF_w)
    SF_wall = Wall_roadSF_fit(SF_road,popt[0],popt[1],popt[2])
    SF_wall[np.isnan(SF_wall)] = np.nanmean(SF_wall)
    SF_wall[np.isnan(SF_wall)] = 0
    return SF_wall

"The block is divided into 25 blocks, this is still oke with the max radius but it does not take to much memory"

"Here we print the info of the run:"
print("gridboxsize is " + str(gridboxsize))
print("max radius is " + str(max_radius))
print("part is 1st up, 2nd left")
print("Data block is HN1")
print("The Date is " + str(Constants.julianday) + " and time is " + str(Constants.hour))
# #
"Switch for 0.5 or 5 m"
# download_directory = config.input_dir_knmi
# SVF_knmi_HN1 = "".join([download_directory, '/SVF_r37hn1.tif'])
# SVF_knmi_HN1 = tf.imread(SVF_knmi_HN1)

grid_ratio = int(gridboxsize/gridboxsize_knmi)
if (gridboxsize==5):
    dtm_HN1 = "".join([input_dir, '/M5_37HN1.TIF'])
    dsm_HN1 = "".join([input_dir, '/R5_37HN1.TIF'])
    [data, data_water] = readdata(minheight,dsm_HN1,dtm_HN1)
    [x_long, y_long] = data.shape

elif (gridboxsize==0.5):
    dtm_HN1 = "".join([input_dir, '/M_37HN1.TIF'])
    dsm_HN1 = "".join([input_dir, '/R_37HN1.TIF'])
    [data, data_water] = readdata(minheight,dsm_HN1,dtm_HN1)
    [x_long, y_long] = data.shape
    #print(data.shape)
    "P1"
    #data = data[:int(x_long/5),:int(y_long/5)]
    "P2"
    data = data[:int(x_long/5),int(y_long/10):int(3*y_long/10)]
    # data_left_lower = data[int(x_long/10):int(3*x_long/10),:int(y_long/5)]
    # data_right_lower = data[int(x_long/10):int(3*x_long/10),int(y_long/10):int(3*y_long/10)]
    #data_left_lower = data[int(x_long/5):int(2*x_long/5),:int(y_long/5)]
    #data_right_lower = data[int(x_long/5):int(2*x_long/5),int(y_long/5):int(2*y_long/5)]
    "P3"
    #data = data[:int(x_long/5),int(2*y_long/5):int(3*y_long/5)]
    [x_len,y_len] = data.shape

"P2: Right Upper, 9 AM"
data_right_upper = data
coords_RU = coordheight(data_right_upper)
[azimuth,el_angle] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,9,radians=True)
SF = reshape_SVF(data_right_upper,coords_RU,gridboxsize,azimuth,el_angle,reshape=False,save_CSV=False,save_Im=False)
print("These are the SFs for right upper, 9am")
print(SF)
"P2: Right Upper, 11 AM"
[azimuth,el_angle] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,11,radians=True)
SF = reshape_SVF(data_right_upper,coords_RU,gridboxsize,azimuth,el_angle,reshape=False,save_CSV=False,save_Im=False)
print("These are the SFs for right upper, 11am")
print(SF)
"P2: Right Upper, 11 AM"
[azimuth,el_angle] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,13,radians=True)
SF = reshape_SVF(data_right_upper,coords_RU,gridboxsize,azimuth,el_angle,reshape=False,save_CSV=False,save_Im=False)
print("These are the SFs for right upper, 1pm")
print(SF)
# with open('SVF_05Matrix.npy', 'rb') as f:
#     SVF_matrix = np.load(f)
# plt.figure()
# plt.imshow(SVF_matrix,vmin=0,vmax=1)
# plt.colorbar()
# plt.figure()
# plt.imshow(SVF_knmi_HN1,vmin=0,vmax=1)
# plt.colorbar()
# plt.show()

# data = data[:int(x_len/4),:int(y_len/4)]
# #data,data_water = MediateData(data,data_water,12.5,12.5,12.5,gridboxsize)
# fig = plt.figure()
# cmap = plt.cm.get_cmap('RdBu')
# cmap.set_bad(color='red')
# [X,Y] = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the surface
# surf = ax.plot_surface(X, Y, data,cmap=cmap)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# # print(data.shape)
# # Add labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Height')
# ax.set_title('3D Surface Plot')

"Height Distribution Dataset"
# data_list = data.flatten()
# data_list = data_list[data_list>0] # if you only want roof surfaces
# print(np.max(data))
# plt.figure()
# plt.hist(data_list, bins= np.linspace(0, 140, 50),weights=np.ones(len(data_list))/len(data_list))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('Height')
# plt.show()

"Colorbars"
# lent = 120
# Zenith = np.linspace(0,np.pi/2,lent,endpoint=False)
# H_w = np.linspace(0.2,5,lent)
# SF_w = np.empty((len(H_w)))
# SF_wall = np.ndarray((len(Zenith),len(H_w)))
# SF_road = np.ndarray((len(Zenith),len(H_w)))
# [h_w,zenith] = np.meshgrid(H_w,Zenith)
# [sf_r,zenith] = np.meshgrid(H_w,Zenith)

# Zenith = [np.pi/6,np.pi/5,np.pi/4,np.pi/3]
# c = ['r','g','b','y']
# Azi,El = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,13,radians=True)
# Zenith = np.pi/2-El
# SF_r_lin = np.linspace(0.01,1,120)
# mat = np.ndarray((len(SF_r_lin),len(Zenith)))
# for i in range(len(Zenith)):
#     for h in range(len(H_w)):
#         [SF_roof, SF_wall[:,h], SF_road[:,h]] = SF_ave_masson(H_w[h],Zenith[i])
#     popt, pcov = curve_fit(f1, SF_road[:,h], SF_wall[:,h])
#
#     mat[:,i] = f1(SF_r_lin,popt[0],popt[1],popt[2])
#
# [zenith,h_w] = np.meshgrid(Zenith,H_w)
# [zenith,sf_r] = np.meshgrid(SF_road,Zenith)

# plt.figure()
# plt.imshow(SF_wall,extent=[min(Zenith),max(Zenith),min(H_w),max(H_w)],vmin=0,vmax=1,aspect='auto')
# plt.colorbar()
# plt.ylabel('Height-over-width ratio [-]')
# plt.xlabel('Zenith angle [deg]')
# plt.figure()
# plt.figure()
# plt.imshow(mat,extent=[min(Zenith),max(Zenith),min(SF_r_lin),max(SF_r_lin)],vmin=0,vmax=1,aspect='auto')
# plt.colorbar()
# plt.ylabel('SF road [0-1]')
# plt.xlabel('Zenith angle [deg]')
# plt.figure()
# plt.imshow(SF_road,extent=[min(H_w),max(H_w),min(Zenith),max(Zenith)],vmin=0,vmax=1,aspect='auto')
# plt.colorbar()
# plt.ylabel('Height over width ratio [-]')
# plt.xlabel('Zenith angle [deg]')
# plt.show()

"We are going to average the data over 12.5m and compute the SVF again"
# coords = coordheight(data)
# blocklength = int(x_len/2*y_len/2)
# SVFs = calc_SVF(coords, max_radius, blocklength, gridboxsize)
# print("The SVFs non averaged")
# print(SVFs)
#
"We now choose a new way of averaging the data:"
# data_new = MediateData(data,data_water,12.5,12.5,12.5,0.5)
# print(data_new.shape)
# gridratio = 25
# coords = coordheight(data_new)
# [x_len, y_len] = data_new.shape
# blocklength = int(x_len/2*y_len/2)
# Azi,El = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,13,radians=True)
# SFs = calc_SF(coords,azi,el,blocklength)
# SVFs = calc_SVF(coords, max_radius, blocklength, int(gridratio*gridboxsize))
# print("The SVFs averaged over 2.5m in x y and z")
# print(SVFs)
# # #
# [data_new, data_water_new] = MediateData(data,data_water,2.5,2.5,2.5,0.5)
# gridratio = 5
# coords = coordheight(data_new)
# [x_len, y_len] = data_new.shape
# blocklength = int(x_len/2*y_len/2)
# SVFs = calc_SVF(coords, max_radius, blocklength, int(gridratio*gridboxsize))
# print("The SVFs with delta x, delta y 12.5m delta z 10m")
# print(SVFs)
# SVF = SVFs5m.SVFs
# SVF = SVFs05m.SVFs
# SVF = SVFGR25.SVFsGR25NMP1
# print(len(SVF))
# np.save('SVFP1_List', SVF)
# SVFs = calc_SVF(coords, max_radius, blocklength, int(gridboxsize*gridratio))
# SF = SF05mHN1.SFs
# SF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SF_matrix[int(coords[i,0]),int(coords[i,1])] = SF[i]
# SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# plt.figure()
# plt.imshow(SF_matrix, vmin=0, vmax=1)
# plt.show()
# np.save('SF1May_aveNM_GR5/SFP1_GR5_NM_18', SF_matrix)

"Height width influence on SVF"
"Let's say you would have a repeating building block of 20m wide and 20 m high with 10 m wide streets: " \
"A repeating infinite canyon." \
"First compute the SVF on this grid"

# for i in range(15):
#     with open('Pickles/1MaySF/SF_may1_'+str(int(i+6))+'_HN1.pickle', 'rb') as f:
#         SF_matrix = pickle.load(f)
#     plt.figure()
#     plt.imshow(SF_matrix,vmin=0,vmax=1)
#     plt.imsave('/Users/rosaliekievits/Desktop/PlaatjesMEP/ShadowFactors_may1/Figures/SF_may1_'+str(int(i+6))+'.png',SF_matrix)

"Movie"
# grid_ratio = 10
# HIST_BINS = np.linspace(0,1,30)
# def getSFdata(hour,gridratio):
#     with open('pickles/1MaySF/SF_may1_'+ str(hour) +'_HN1.pickle', 'rb') as f:
#         SF_matrix = pickle.load(f)
#     [SF_roof,SF_road] = average_surfacetype(SF_matrix,data,gridratio)
#     return SF_road
#
# import matplotlib.animation as animation

# frames = [] # for storing the generated images
# fig = plt.figure()
# for i in range(15):
#     # with open('pickles/1MaySF/SF_may1_'+ int(i+6) +'_HN1.pickle', 'rb') as f:
#     #     SF_matrix = pickle.load(f)
#     with open('SF1May_aveNM_GR5/SFP1_GR5_NM_' + str(int(i+6)) + '.npy', 'rb') as f:
#         SF_matrix = np.load(f)
#     frames.append([plt.imshow(SF_matrix,animated=True)])
# ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=1000)
# writergif = animation.PillowWriter(fps=30)
# ani.save('/Users/rosaliekievits/Desktop/PlaatjesMEP/ShadowFactors_may1/Out files May 1/New/SFmovie_aveNM.gif', writer=writergif)
# plt.show()
"end movie"

"Shadowfactor"

# [azimuth,el_angle] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,5,radians=True)
# SF = reshape_SVF(data,coords,gridboxsize,azimuth,el_angle,reshape=False,save_CSV=False,save_Im=False)
# print(SF)
# # gridratio = 25
# SF = SF05mHN1.SFs
# "The sun rise and sunset time for 1 may are 6 and 20 o clock (GMT)"
# SF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SF_matrix[int(coords[i,0]),int(coords[i,1])] = SF[i]
# SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# plt.figure()
# plt.imshow(SF_matrix, vmin=0, vmax=1, aspect='auto')
# plt.show()
# with open('pickles/1MaySF/SF_may1_18_HN1.pickle', 'wb') as f:
#     pickle.dump(SF_matrix, f)
"Shadowfactor for 24 hours"
"Don't forget to comment out import SVFs05 !! and change hours"

"end of Shadowfactor for 24 hours"


"Save all SF, areafractions and SVF to pickles"
#coords = coordheight(data)
# SVF = SVFs05m.SVFs
# plt.figure()
# plt.hist(SVF, bins = 30,weights=np.ones(len(SVF))/len(SVF))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF [0-1]')
# plt.show()
# gridratio = 25
# SF = SF05mHN1.SFs
# SVF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SVF_matrix[int(coords[i,0]),int(coords[i,1])] = SVF[i]
# SVF_matrix = SVF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# np.save('SVF_05Matrix', SVF_matrix)
# with open('SVF_GR5Matrix.npy', 'rb') as f:
#     SVF_matrix = np.load(f)
# plt.figure()
# plt.imshow(SVF_matrix, vmin=0, vmax=1)
# plt.show()
#data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# SVF = SVFs5m.SVFs
# gridratio = 5
# data = average_svf(data,gridratio)
# [x_len,y_len] = data.shape
# data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# print(data.shape)
#coords = coordheight(data,gridboxsize)
# plt.figure()
# plt.hist(SVF, bins = 30,weights=np.ones(len(SVF))/len(SVF))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF [0-1]')
# plt.show()
# SVF_roof = []
# SVF_road = []
# with open('SVF_GR5Matrix.npy', 'rb') as f:
#     SVF_matrix = np.load(f)
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         if data[i,j]>0:
#             SVF_roof.append(SVF_matrix[i,j])
#         elif data[i,j]<=0:
#             SVF_road.append(SVF_matrix[i,j])
# print(len(SVF_road))
# print(len(SVF_roof))
# print(len(SVF_road)/(len(SVF_roof)+len(SVF_road)))
# plt.figure()
# plt.hist(SVF_roof, bins = 30,weights=np.ones(len(SVF_roof))/len(SVF_roof))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF roof [0-1]')
# plt.figure()
# plt.hist(SVF_road, bins = 30,weights=np.ones(len(SVF_road))/len(SVF_road))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF road [0-1]')
# plt.show()
# SVF_w = Fit_WallvsRoadMasson(SVF_road)
# plt.figure()
# plt.hist(SVF_w, bins = 30,weights=np.ones(len(SVF_w))/len(SVF_w))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF wall [0-1]')
# plt.show()
# SF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SF_matrix[int(coords[i,0]),int(coords[i,1])] = SF[i]
# SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# with open('pickles/1MaySF/SF_may1_17_HN1.pickle', 'wb') as f:
#     pickle.dump(SF_matrix, f)
# plt.figure()
# plt.imshow(SF_matrix, vmin=0, vmax=1, aspect='auto')
# plt.show()

# gridratio = 25
# SFs = SF05mHN1.SFs
# eps = 0.2
# coords = coordheight(data)
# SF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SF_matrix[int(coords[i,0]),int(coords[i,1])] = SFs[i]
# SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# # plt.figure()
# # plt.imshow(SF_matrix, vmin=0, vmax=1, aspect='auto')
# # plt.show()


"MaxR / stepsbeta / minheight parameter variations"
# max_radius = 100
# minheight = np.linspace(0,5,11)
# #steps_beta = [60,90,120,180,240,270,360,540,720]
# SVFp1 = np.empty((len(minheight)))
# SVFp2 = np.empty((len(minheight)))
# SVFp3 = np.empty((len(minheight)))
# for m in range(len(minheight)):
#     dtm_HN1 = "".join([input_dir, '/M_37HN1.TIF'])
#     dsm_HN1 = "".join([input_dir, '/R_37HN1.TIF'])
#     [data, data_water] = readdata(minheight[m],dsm_HN1,dtm_HN1)
#     [x_long, y_long] = data.shape
#     data = data[:int(x_long/5),:int(y_long/5)]
#     [x_long, y_long] = data.shape
#     coords = coordheight(data)
#     p1 = [625, 500, 0]
#     p2 = [1250, 1000, 8.65600014]
#     #p2 = [625, 1500, data[625, 1500]]
#     #print(p2)
#     p3 = [1875, 1500, 13.48999977]
#     #p3 = [1875, 500, data[1875, 500]]
#     #print(p3)
#     SVFp1[m] = SkyViewFactor(p1,coords,max_radius,gridboxsize,steps_beta)
#     SVFp2[m] = SkyViewFactor(p2,coords,max_radius,gridboxsize,steps_beta)
#     SVFp3[m] = SkyViewFactor(p3,coords,max_radius,gridboxsize,steps_beta)
#     # SVFp1inf = SkyViewFactor(p1,coords,max_radius,gridboxsize,720)
#     # SVFp2inf = SkyViewFactor(p2,coords,max_radius,gridboxsize)
#     # SVFp3inf = SkyViewFactor(p3,coords,max_radius,gridboxsize)
# plt.figure()
# plt.imshow(data,vmin=0,vmax=50)
# plt.figure()
# SVFp1_rel = SVFp1/SVFp1[0]
# SVFp2_rel = SVFp2/SVFp2[0]
# SVFp3_rel = SVFp3/SVFp3[0]
# "Plot the svf relative to the final SVF!!"
# plt.plot(minheight,SVFp1,label='P1')
# plt.plot(minheight,SVFp2,label='P2')
# plt.plot(minheight,SVFp3,label='P3')
# print(SVFp1)
# print(SVFp2)
# print(SVFp3)
# plt.xlabel('Minimum Height [m]')
# plt.ylabel('SVF [0-1]') # relative to the SVF with 0 minimum height
# plt.legend()
# plt.show()
"end"

"End of Mean SF per surface type"

"Time elapsed"
endtime = time.time()
elapsed_time = endtime-sttime
print('Execution time:', elapsed_time, 'seconds')

