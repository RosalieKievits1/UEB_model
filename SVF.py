import numpy as np
import matplotlib.pyplot as plt
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
def coordheight(data,gridboxsize):
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
    if (gridboxsize==5):
        rowcount_block = int((x_len-2*max_radius/gridboxsize)*(y_len-2*max_radius/gridboxsize))
        for i in range(x_len):
            for j in range(y_len):
                """we need to make a list of coordinates where the center block is first"""
                if ((max_radius/gridboxsize)<=i and i<(x_len-max_radius/gridboxsize) and (max_radius/gridboxsize)<=j and j<(y_len-max_radius/gridboxsize)):
                    coords[rowcount_center,0] = i
                    coords[rowcount_center,1] = j
                    coords[rowcount_center,2] = data[i,j]
                    rowcount_center += 1
                elif (i<(max_radius/gridboxsize) or i>=(x_len-max_radius/gridboxsize) or j<(max_radius/gridboxsize) or j>=(y_len-max_radius/gridboxsize)):
                    coords[rowcount_block,0] = i
                    coords[rowcount_block,1] = j
                    coords[rowcount_block,2] = data[i,j]
                    rowcount_block += 1
    elif (gridboxsize==0.5):
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

def SkyViewFactor(point, coords, max_radius,gridboxsize):
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
    SVF = np.around((np.sum(np.cos(betas)**2)/steps_beta),3)
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
    if np.count_nonzero(coords[np.logical_and((np.logical_and((angles > beta_min), (angles < beta_max))), ((np.tan(elevation_angle)*radii)<(coords[:,2]-point[2]))),:])>0:
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
    #blocklength = int((x_len-2*max_radius/gridboxsize)*(y_len-2*max_radius/gridboxsize))
    "Compute SVF and SF and Reshape the shadow factors and SVF back to nd array"
    #SVFs = calc_SVF(coords, max_radius, blocklength, gridboxsize)
    #print(SVFs)
    SFs = calc_SF(coords,azimuth,zenith,blocklength)
    "If reshape is true we reshape the arrays to the original data matrix"
    if (reshape == True) & (SFs is not None):
        #SVF_matrix = np.ndarray([x_len-2*max_radius/gridboxsize,y_len-2*max_radius/gridboxsize])
        #SVF_matrix = np.ndarray([x_len,y_len])
        SF_matrix = np.ndarray([x_len,y_len])
        for i in range(blocklength):
            #SVF_matrix[coords[i,0]-max_radius/gridboxsize,coords[i,1]-max_radius/gridboxsize] = SVFs[i]
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


def geometricProperties(data,grid_ratio,gridboxsize):
    """
    Function that determines the average height over width of an area,
    the average height over width, and the built fraction of an area
    :param data: height data of city
    :return:
    H_W : height over width ratio
    ave_height : average height of the area
    delta: fraction of built area
    """
    [x_long, y_long] = data.shape
    # x_long = int(x_long/2)
    # y_long = int(y_long/2)
    [Wall_area, wall_area_total] = wallArea(data,gridboxsize)
    Wall_area_gridcell = np.sum(Wall_area,axis=2)
    delta = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    ave_height = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    road_elements = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    built_elements = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    water_elements = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    wall_area_med = np.zeros([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(int(x_long/grid_ratio)):
        for j in range(int(y_long/grid_ratio)):
            part = data[i*grid_ratio:(i+1)*grid_ratio, j*grid_ratio:(j+1)*grid_ratio]
            part_wall = Wall_area_gridcell[i*grid_ratio:(i+1)*grid_ratio, j*grid_ratio:(j+1)*grid_ratio]
            ave_height[i,j] = np.mean(part[part>0])
            road_elements[i,j] = np.count_nonzero(part==0)
            built_elements[i,j] = np.count_nonzero(part>0)
            "The road elements are actually also water elements"
            water_elements[i,j] = np.count_nonzero(part==-1)
            delta[i,j] = built_elements[i,j]/(road_elements[i,j]+built_elements[i,j]+water_elements[i,j])
            wall_area_med[i,j] = np.sum(part_wall)

    """We want to determine the wall area from the height and delta
    Say each block is a separate building: then the wall area would be 4*sum(builtarea), 
    but since we have a certain density of houses we could make a relation 
    between density and buildings next to each other"""
    Roof_area = built_elements*gridboxsize**2
    Road_area = road_elements*gridboxsize**2
    Water_area = water_elements*gridboxsize**2

    Total_area = Roof_area + wall_area_med + Road_area + Water_area
    Roof_area = Roof_area + Water_area
    """Fractions of the area of the total surface"""
    Roof_frac = np.around(Roof_area/Total_area,3)
    Wall_frac = np.around(wall_area_med/Total_area,3)
    Road_frac = np.around(Road_area/Total_area,3)
    Water_frac = np.around(Water_area/Total_area,3)
    #Road_frac = Road_frac + Water_frac
    H_W = ave_height * delta
    return Roof_frac, Wall_frac, Road_frac #Roof_area, wall_area_med, Road_area#

def average_svf_surfacetype(SVF_matrix,data, grid_ratio):
    [x_long, y_long] = SVF_matrix.shape
    SVF_ave_roof = np.ndarray([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    SVF_ave_road = np.ndarray([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(int(x_long/grid_ratio)):
        for j in range(int(y_long/grid_ratio)):
            data_part = data[int(i*grid_ratio):int((i+1)*grid_ratio), int(j*grid_ratio):int((j+1)*grid_ratio)]
            part = SVF_matrix[int(i*grid_ratio):int((i+1)*grid_ratio), int(j*grid_ratio):int((j+1)*grid_ratio)]
            SVF_ave_roof[i,j] = np.mean(part[data_part>0])
            SVF_ave_road[i,j] = np.mean(part[data_part<=0])
    return SVF_ave_roof,SVF_ave_road

def average_svf(SVF_matrix, grid_ratio):
    [x_long, y_long] = SVF_matrix.shape
    SVF_ave = np.ndarray([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(int(x_long/grid_ratio)):
        for j in range(int(y_long/grid_ratio)):
            part = SVF_matrix[int(i*grid_ratio):int((i+1)*grid_ratio), int(j*grid_ratio):int((j+1)*grid_ratio)]
            SVF_ave[i,j] = np.mean(part)
    return SVF_ave

def SF_wall(point,coords,type,wall_area,azimuth,elevation_angle):
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
    beta_min = np.asarray(- np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)
    beta_max = np.asarray(np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)

    if np.logical_or((np.count_nonzero(coords[np.logical_and((np.logical_and((angles > beta_min), (angles < beta_max))), ((np.tan(elevation_angle)*radii)<(coords[:,2]-point[2]))),:])>0),type==0):
        Shadowfactor = 0
    else:
        Shadowfactor = 1
    """in all other cases there is no point in the same direction as the sun that is higher
    so the shadowfactor is 1: the point receives radiation"""
    return Shadowfactor

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
    if gridboxsize==5:
        wall_area = np.ndarray([int(x_len-2*max_radius/gridboxsize),int(y_len-2*max_radius/gridboxsize),4])
    elif gridboxsize==0.5:
        #wall_area = np.ndarray([int(x_len/2),int(y_len/2),4])
        wall_area = np.zeros([int(x_len),int(y_len),4])

    if (gridboxsize == 0.5):
        for i in range(int(x_len/4),int(3*x_len/4)):
            for j in range(int(y_len/4),int(3*y_len/4)):
                if (data[i,j]>0):
                    """We check for all the points surrounding the building if they are also buildings, 
                    if the building next to it is higher the wall belongs to the building next to it,
                    if the current building is higher, the exterior wall is the difference in height * gridboxsize"""
                    # wall_area[int(i-x_len/4),int(j-y_len/4),0] = max(data[i,j]-data[i-1,j],0)*gridboxsize
                    # wall_area[int(i-x_len/4),int(j-y_len/4),1] = max(data[i,j]-data[i,j+1],0)*gridboxsize
                    # wall_area[int(i-x_len/4),int(j-y_len/4),2] = max(data[i,j]-data[i+1,j],0)*gridboxsize
                    # wall_area[int(i-x_len/4),int(j-y_len/4),3] = max(data[i,j]-data[i,j-1],0)*gridboxsize
                    wall_area[i,j,0] = max(data[i,j]-data[i-1,j],0)*gridboxsize
                    wall_area[i,j,1] = max(data[i,j]-data[i,j+1],0)*gridboxsize
                    wall_area[i,j,2] = max(data[i,j]-data[i+1,j],0)*gridboxsize
                    wall_area[i,j,3] = max(data[i,j]-data[i,j-1],0)*gridboxsize
                    """The wall area corresponding to that building is"""
                    #wall_area[int(i-x_len/4),int(j-y_len/4)] = wall1+wall2+wall3+wall4
                elif (data[i,j]==0):
                    wall_area[int(i-x_len/4),int(j-x_len/4),:] = 0
    elif (gridboxsize==5):
        i = int(max_radius/gridboxsize)
        j = int(max_radius/gridboxsize)
        while i < int(x_len-max_radius/gridboxsize):
            while j < int(y_len-max_radius/gridboxsize):
                if (data[i,j]>0):
                    """We check for all the points surrounding the building if they are also buildings, 
                    if the building next to it is higher the wall belongs to the building next to it,
                    if the current building is higher, the exterior wall is the difference in height * gridboxsize"""
                    wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize),0] = max(data[i,j]-data[i-1,j],0)*gridboxsize
                    wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize),2] = max(data[i,j]-data[i,j+1],0)*gridboxsize
                    wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize),1] = max(data[i,j]-data[i+1,j],0)*gridboxsize
                    wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize),3] = max(data[i,j]-data[i,j-1],0)*gridboxsize
                    """The wall area corresponding to that building is"""
                    #wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize)] = np.sum(wall,axis=2)
                elif (data[i,j]==0):
                    wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize),:] = 0
                j+=1
            i+=1
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

    num_slices = int(np.rint(wall_len*2))
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
        psi_ave = np.mean(psi)
        """Where the index of betas fall within the min and max beta, and there is not already a larger psi blocking"""
        betas[np.nonzero(np.logical_and(betas<psi_ave,(np.logical_and((beta_min <= beta_lin),(beta_lin < beta_max)))))] = psi_ave
        #betas_zero[np.nonzero(np.logical_and(betas_zero<psi_zero,(np.logical_and((beta_min <= beta_lin),(beta_lin < beta_max)))))] = psi_zero
    #print(betas)
        # if dome_zero[d,2]==0:
        #     closest[np.nonzero(np.logical_and((closest > dome_zero[d,3]), np.logical_and((beta_min <= beta_lin), (beta_lin < beta_max))))] = dome_zero[d,3]
    SVF_wall = np.around((np.sum(np.cos(betas)**2)/steps_beta),3)
    return SVF_wall

def compute_wvf(data):
    """
    Compute the wall view factors (WVF) for all points on the DSM with external walls.

    Parameters:
    - dsm (2D numpy array): digital surface model
    - max_distance (float): maximum distance to consider for the calculations
    - num_divisions (int): number of divisions in the azimuth angle
    - num_slices (int): number of slices along the wall height to consider
    - num_workers (int): number of worker processes to use for multiprocessing

    Returns:
    - 2D numpy array: wall view factors (WVF) for each point
    """
    [x_len,y_len] = data.shape
    [walls_matrix,total_wall_area] = wallArea(data,gridboxsize)
    WVF = np.zeros([walls_matrix.shape])

    def parallel_runs_SVF_wall():
        points = [coords[i,:] for i in range(blocklength)]
        pool = Pool()
        SVF_list_north = []
        SVF_list_east = []
        SVF_list_south = []
        SVF_list_west = []
        wall_len_n = walls_matrix[points[0],points[1],0]
        wall_len_e = walls_matrix[points[0],points[1],1]
        wall_len_s = walls_matrix[points[0],points[1],2]
        wall_len_w = walls_matrix[points[0],points[1],3]
        SVF_par_north = partial(SVF_wall, coords=coords,maxR=max_radius,type=0,wall_len=wall_len_n) # prod_x has only one argument x (y is fixed to 10)
        SVF_par_east = partial(SVF_wall, coords=coords,maxR=max_radius,type=1,wall_len=wall_len_e) # prod_x has only one argument x (y is fixed to 10)
        SVF_par_south = partial(SVF_wall, coords=coords,maxR=max_radius,type=2,wall_len=wall_len_s) # prod_x has only one argument x (y is fixed to 10)
        SVF_par_west = partial(SVF_wall, coords=coords,maxR=max_radius,type=3,wall_len=wall_len_w) # prod_x has only one argument x (y is fixed to 10)
        SVF_n = pool.map(SVF_par_north, points)
        SVF_e = pool.map(SVF_par_east, points)
        SVF_s = pool.map(SVF_par_south, points)
        SVF_w = pool.map(SVF_par_west, points)
        pool.close()
        pool.join()
        if SVF is not None:
            SVF_list_north.append(SVF_n)
            SVF_list_east.append(SVF_e)
            SVF_list_south.append(SVF_s)
            SVF_list_west.append(SVF_w)
        return SVF_list_north, SVF_list_east, SVF_list_south, SVF_list_west

    if __name__ == '__main__':
        result = parallel_runs_SVF_wall()
        return result

    return SVF
""""""


"The block is divided into 25 blocks, this is still oke with the max radius but it does not take to much memory"

"Here we print the info of the run:"
print("gridboxsize is " + str(gridboxsize))
print("max radius is " + str(max_radius))
print("part is 1st up, 1st left")
print("Data block is HN1")
print("The Date is " + str(Constants.julianday) + " and time is " + str(Constants.hour))
# #
"Switch for 0.5 or 5 m"
# download_directory = config.input_dir_knmi
# SVF_knmi_HN1 = "".join([download_directory, '/SVF_r37hn1.tif'])
# SVF_knmi_HN1 = tf.imread(SVF_knmi_HN1)
#
# grid_ratio = int(gridboxsize/gridboxsize_knmi)
if (gridboxsize==5):
    dtm_HN1 = "".join([input_dir, '/M5_37HN1.TIF'])
    dsm_HN1 = "".join([input_dir, '/R5_37HN1.TIF'])
    data = readdata(minheight,dsm_HN1,dtm_HN1)
    [x_long, y_long] = data.shape

elif (gridboxsize==0.5):
    dtm_HN1 = "".join([input_dir, '/M_37HN1.TIF'])
    dsm_HN1 = "".join([input_dir, '/R_37HN1.TIF'])
    data = readdata(minheight,dsm_HN1,dtm_HN1)
    [x_long, y_long] = data.shape
    data = data[:int(x_long/5),:int(y_long/5)]
    #SVF_knmi_HN1 = SVF_knmi_HN1[:int(x_long/5),:int(y_long/5)]
    [x_len,y_len] = data.shape


"Shadowfactor"
# coords = coordheight(data,gridboxsize)
# [azimuth,el_angle,T_ss,T_sr] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,Constants.hour,radians=True)
# SF = reshape_SVF(data,coords,gridboxsize,azimuth,el_angle,reshape=False,save_CSV=False,save_Im=False)
# print(SF)
"The sun rise and sunset time for 1 may are 6 and 20 o clock (GMT)"
# SF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SF_matrix[int(coords[i,0]),int(coords[i,1])] = SF[i]
# SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# print(SF_matrix)

"Shadowfactor for 24 hours"
"Don't forget to comment out import SVFs05 !! and change hours"
coords = coordheight(data,gridboxsize)
hours = np.linspace(8,12,5)
SF_matrix = np.ndarray([int(x_len),int(y_len)])
for h in range(len(hours)):
    [azimuth,el_angle,T_ss,T_sr] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,hours[h],radians=True)
    SFs = reshape_SVF(data,coords,gridboxsize,azimuth,el_angle,reshape=False,save_CSV=False,save_Im=False)
    print("The Date is " + str(Constants.julianday) + " and time is " + str(hours[h]))
    print(SFs)
#     for i in range(int(x_len/2*y_len/2)):
#         SF_matrix[int(coords[i,0]),int(coords[i,1])] = SF[i]
#     SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#     print(SF_matrix)
    #np.savetxt("SFmatrix" + str(hours[h]) + ".csv", SF_matrix, delimiter=",")
"end of Shadowfactor for 24 hours"

"Save all SF, areafractions and SVF to pickles"
# coords = coordheight(data,gridboxsize)
# gridratio = 25
# SF = SF05mHN1.SFs
# # # SVF_matrix = np.ndarray([x_len,y_len])
# # # for i in range(int(x_len/2*y_len/2)):
# # #     SVF_matrix[int(coords[i,0]),int(coords[i,1])] = SVF[i]
# # # SVF_matrix = SVF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# SF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SF_matrix[int(coords[i,0]),int(coords[i,1])] = SF[i]
# SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# "SVF loop for different grid ratios"
# gridratio = [5,10,25,50,125,250]
# mean_roof = np.ndarray((len(gridratio),1))
# roof_p1 = np.ndarray((len(gridratio),1))
# roof_p2 = np.ndarray((len(gridratio),1))
# frac_roof_p1 = np.ndarray((len(gridratio),1))
# frac_roof_p2 = np.ndarray((len(gridratio),1))
# frac_roof_mean = np.ndarray((len(gridratio),1))
#
#
# plt.figure()
# for i in range(len(gridratio)):
#     Roof_frac, Wall_frac, Road_frac = geometricProperties(data,gridratio[i],gridboxsize)
#     [SF_roof,SF_road] = average_svf_surfacetype(SF_matrix,data,gridratio[i])
#     #data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#     SF_wall = 1-average_svf(SF_matrix,gridratio[i])
#     mean_roof[i] = np.nanmean(SF_roof)#*np.mean(Roof_frac)
#     roof_p1[i] = SF_roof[0,-1]#np.nanmin(SF_wall)
#     roof_p2[i] = SF_roof[-1,-1]#np.nanmax(SF_wall)
#     frac_roof_p2[i] = Roof_frac[0,-1]
#     frac_roof_p1[i] = Roof_frac[0,-1]
#     frac_roof_mean[i] = np.nanmean(Roof_frac)
#
# plt.plot(gridratio, mean_wall,'r',label='Mean SF')
# plt.plot(gridratio, wall_p1,'b',label='SF p1')
# plt.plot(gridratio, wall_p2,'y',label='SF p2')
# plt.plot(gridratio, frac_wall_mean,'r',linestyle='dotted',label='Mean Area Fraction')
# plt.plot(gridratio, frac_wall_p1,'b',linestyle='dotted',label='Area Fraction')
# plt.plot(gridratio, frac_wall_p2,'y',linestyle='dotted',label='Area Fraction p2')
# #plt.plot(gridratio, mean_road,label='Mean Road SVF')
# plt.xlabel('gridratio')
# plt.legend(loc='upper right')
# plt.ylabel('SF [0-1]')
# plt.show()

# gridratio = 25
# SFs = SF05mHN1.SFs
# coords = coordheight(data,gridboxsize)
# SF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SF_matrix[int(coords[i,0]),int(coords[i,1])] = SFs[i]
# SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# plt.figure()
# plt.imshow(SF_matrix, vmin=0, vmax=1, aspect='auto')
# plt.show()
# [SF_roof,SF_road] = average_svf_surfacetype(SF_matrix,data,gridratio)
# Roof_frac, Wall_frac, Road_frac = geometricProperties(data,gridratio,gridboxsize)
# SF_wall = (1-average_svf(SF_matrix,gridratio))/Wall_frac
# SF_wall[Wall_frac<eps] = 0
# SF_road[Road_frac==0] = 0
# SF_roof[Roof_frac==0] = 0
# SVF = SVFs05m.SVFs
# SVF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SVF_matrix[int(coords[i,0]),int(coords[i,1])] = SVF[i]
# SVF_matrix = SVF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# [SVF_roof,SVF_road] = average_svf_surfacetype(SVF_matrix,data,gridratio)
# data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# SVF_wall = 1-average_svf(SVF_matrix,gridratio)
# SVF_roof[Roof_frac == 0] = 0 #nan=np.nanmean(SVF_roof))
# SVF_road[Road_frac == 0] = 0 #nan=np.nanmean(SVF_road))

# "Pickle the area fractions"
# with open('pickles/roofFrac25_HN1.pickle', 'wb') as f:
#     pickle.dump(Roof_frac, f)
# with open('pickles/wallFrac25_HN1.pickle', 'wb') as f:
#     pickle.dump(Wall_frac, f)
# with open('pickles/roadFrac25_HN1.pickle', 'wb') as f:
#     pickle.dump(Road_frac, f)
# "Pickle the shadow factors"
# with open('pickles/RoofSF_may1_' +str(Constants.hour) + '_25_HN1.pickle', 'wb') as f:
#     pickle.dump(SF_roof, f)
# with open('pickles/WallSF_may1_' +str(Constants.hour) + '_25_HN1.pickle', 'wb') as f:
#     pickle.dump(SF_wall, f)
# with open('pickles/RoadSF_may1_' +str(Constants.hour) + '_25_HN1.pickle', 'wb') as f:
#     pickle.dump(SF_road, f)
# "Pickle the Sky view factors"
# with open('pickles/RoofSVF_25_HN1.pickle', 'wb') as f:
#     pickle.dump(SVF_roof, f)
# with open('pickles/WallSVF_25_HN1.pickle', 'wb') as f:
#     pickle.dump(SVF_wall, f)
# with open('pickles/RoadSVF_25_HN1.pickle', 'wb') as f:
#     pickle.dump(SVF_road, f)
#
# # plt.figure()
# # #plt.subplot(1, 2, 1)
# # plt.imshow(SF_matrix, vmin=0, vmax=1, aspect='auto')
# # plt.show()
#
# # print(SF_roof)
# # print(SF_road)

"Compare the Sky view factor of masson for an urban geometry"
"the road is 10 data points wide, "
# widths = np.linspace(10,15,6)/gridboxsize
# SFs = np.ndarray((len(widths),1))
# SF_w = np.ndarray((len(widths),1))
# H_w = np.ndarray((len(widths),1))
# #sf_masson_roads = np.ndarray((len(widths),1))
# phi_masson_roads = np.ndarray((len(widths),1))
# #sf_masson_walls= np.ndarray((len(widths),1))
# height_r = 15
# height_l = 15
# len_mat = 500 # shape of the matrix
# plt.figure()
# gridratio = 500
# phi_masson_walls = np.ndarray((len(widths),1))
# SVFs = np.ndarray((len(widths),1))
# SVF_roofs = np.ndarray((len(widths),1))
# SVFs_w = np.ndarray((len(widths),1))
# SVFs_w_2 = np.ndarray((len(widths),1))
# #el_angles = [5*np.pi/12,5*np.pi/15,5*np.pi/18]
# colours = ['r','b','y']
# # for e in range(len(el_angles)):
# #     elevation_angle = el_angles[e]
# for i in range(len(widths)):
#     int(widths[i])
#     elevation_angle = np.pi/3
#     h_w = height_l/(widths[i]*gridboxsize)#*gridboxsize)
#     zenith = np.pi/2-elevation_angle
#     lamb_zero = np.arctan(1/h_w)
#     H_w[i] = h_w
#     ucm_matrix = np.ones((len_mat,len_mat))*height_l
#     ucm_matrix[:,int(len_mat/2-widths[i]/2):int(len_mat/2+widths[i]/2)] = 0
#     ucm_matrix[:,int(len_mat/2+widths[i]/2)+1::] = height_r
#     coords = coordheight(ucm_matrix,gridboxsize)
#     # [Roof_frac, Wall_frac, Road_frac] = geometricProperties(ucm_matrix,gridratio,gridboxsize)
#     # SF_matrix = np.ones((ucm_matrix.shape))
#     phi_masson_roads[i] = np.sqrt((h_w**2+1))-h_w
#     phi_masson_walls[i] = 1/2*(h_w+1-np.sqrt(h_w**2+1))/h_w
#     #SF = np.ndarray((int(widths[i]),1))
#     #SVF_w = np.ndarray((int(widths[i]),1))
#     SVF = np.ndarray((int(widths[i]),1))
#         # if (zenith > lamb_zero):
#         #     sf_masson_walls[i] = 1/2/h_w
#         #     sf_masson_roads[i] = 0
#         # elif (zenith < lamb_zero):
#         #     sf_masson_walls[i] = 1/2*np.tan(zenith)
#         #     sf_masson_roads[i] = 1-h_w*np.tan(zenith)
#     # for j in range(int(widths[i])):
#     #     point = [len_mat/2,len_mat/2-widths[i]/2+j,0]
#         # SF[j] = shadowfactor(point,coords,np.pi/2,elevation_angle)
#         # SF_matrix[:,int(len_mat/2-widths[i]/2+j)] = SF[j]
#         #SVF[j] = SkyViewFactor(point,coords,max_radius,gridboxsize)
#         #print(Wall_frac)
#         # SFs[i] = np.mean(SF)
#         # SF_w[i] = (1-np.mean(SF_matrix))/np.mean(2*Wall_frac)
#     #point_roof_l = [len_mat/2,len_mat/2-widths[i]/2-1,height_l]
#     p_wall_2 = [len_mat/2,len_mat/2+widths[i]//2,height_r]
#     p_wall = [len_mat/2,len_mat/2-widths[i]/2-1,height_l]
#     #SVF_roofs[i] = SkyViewFactor(point_roof_l,coords,max_radius,gridboxsize)
#     SVFs_w[i] = SVF_wall(p_wall,coords,max_radius,1,height_l)
#     SVFs_w_2[i] = SVF_wall(p_wall_2,coords,max_radius,3,height_r)
#     #SVFs[i] = np.mean(SVF)
#     # plt.plot(H_w,SF_w,colours[e],label='Numerical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
#     # plt.plot(H_w,sf_masson_walls,colours[e],linestyle='dotted',label='Analytical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
#     # print(SF_w)
#     # print(sf_masson_walls)
# print(SVFs_w)
# print(SVFs_w_2)
# print(phi_masson_walls)
# plt.plot(H_w,phi_masson_walls,'blue',linestyle='dotted',label='Analytical')
# plt.plot(H_w,SVFs_w,'blue',label='Left Wall')
# plt.plot(H_w,SVFs_w_2,'lightblue',label='Right Wall ')
# #plt.plot(widths,SVFs,'red',label='Analytical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
# #plt.plot(widths,SVFs,'red',label='Analytical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
#
# plt.xlabel('Width ratio')
# plt.ylabel('SVF [0-1]')
# plt.ylim((0,1))
# plt.legend(loc='upper right')
# plt.show()
"end"



# with open('pickles/SVF_matrix05m.pickle', 'rb') as f:
#     SVF_matrix = pickle.load(f)

"Shadowfactor for 1 day in may for different surfaces"
# plt.figure()
# gridratio = 25
# hours = np.linspace(8,15,8)
# meanSFroof = np.ndarray((len(hours),1))
# meanSFroad = np.ndarray((len(hours),1))
# meanSFwall = np.ndarray((len(hours),1))
# for h in range(len(hours)):
#     with open('pickles/RoofSF_may1_' + str(hours[h]) +'_25_HN1.pickle', 'rb') as f:
#         SF_matrix_roof = pickle.load(f)
#     with open('pickles/WallSF_may1_' + str(hours[h]) +'_25_HN1.pickle', 'rb') as f:
#         SF_matrix_wall = pickle.load(f)
#     with open('pickles/RoadSF_may1_' + str(hours[h]) +'_25_HN1.pickle', 'rb') as f:
#         SF_matrix_road = pickle.load(f)
#     meanSFroof[h] = np.nanmean(SF_matrix_roof)
#     meanSFwall[h] = np.nanmean(SF_matrix_wall)
#     meanSFroad[h] = np.nanmean(SF_matrix_road)
#
# plt.plot(hours, meanSFroof,'r',label='Roof')
# plt.plot(hours, meanSFwall,'b',label='Wall')
# plt.plot(hours, meanSFroad,'y',label='Road')
# plt.xlabel('time [GMT, hour]')
# plt.legend(loc='upper right')
# plt.ylabel('Mean SF [0-1]')
# plt.show()
"End of Mean SF per surface type"

# [wall_matrix,totalwall] = wallArea(data,gridboxsize)
# wall = wall_matrix[int(x_len/4),int(y_len/4),2]
# print(SVF_WVF_wall(point,coords,max_radius,2,wall,10))


# SVF_wall[SVF_wall < 0] = 0
# SVF_wall[Wall_area == 0] = 0
#
# SVF_wall = np.nan_to_num(SVF_wall, nan=np.nanmean(SVF_wall))

# SVF_wall = np.nan_to_num(SVF_wall, nan=0)

# print(SVF_wall)
# print(Road_frac)
# print(Roof_frac)
# WVF_roof = 1-SVF_roof
# WVF_road = 1-SVF_road
# GVF_wall = WVF_road*(Road_frac/(Wall_frac+Road_frac+Roof_frac))
# RVF_wall = WVF_roof*(Roof_frac/(Wall_frac+Roof_frac+Road_frac))
# RVF_wall = np.nan_to_num(RVF_wall, nan=0)
# #GVF_wall = np.nan_to_num(GVF_wall, nan=0)
#
# #RVF_wall = np.nan_to_num(RVF_wall, nan=0) #nan=np.nanmean(RVF_wall))
# WVF_wall = 1-SVF_wall-GVF_wall-RVF_wall
# WVF_wall[WVF_wall<0] =0
# print("GVF_wall")
# print(np.mean(GVF_wall))
# print(np.max(GVF_wall))
# print(np.min(GVF_wall))
#
# #GVF_wall = np.nan_to_num(GVF_wall, nan=np.nanmean(GVF_wall))
#
# print("RVF")
#WVF_wall = np.nan_to_num(WVF_wall, nan=np.nanmean(WVF_wall))
#print(np.count_nonzero(np.logical_and(Roof_area==0,Wall_area==0)))
# Since the wall reflects on the roof, the roof should also reflect on the wall??

#RVF_wall = np.nan_to_num(RVF_wall, nan=np.nanmean(RVF_wall))
# print(np.mean(RVF_wall))
# #WVF_wall[WVF_wall<0] = 0
# # print(Wall_frac)
# # print(np.max(Wall_frac))
# # print(np.min(Wall_frac))
# print(np.max(RVF_wall))
# print(np.min(RVF_wall))
# print("WVF_wall")
# print(np.mean(WVF_wall))
# print(np.max(WVF_wall))
# print(np.min(WVF_wall))
"Now we have a SVF for roof and road surfaces averaged over 5m gridcells, " \
"and the data averaged over 5m, surface fractions for now"

# plt.subplot(1, 2, 2)
# plt.imshow(SVF_knmi_HN1,vmin=0, vmax = 1, aspect='auto')
# #plt.colorbar()
# plt.show()
# print(SVFs05[50:100])
# meanSVFs05 = sum(SVFs05)/len(SVFs05)
# print('The mean of the SVFs computed on 0.5m is ' + str(meanSVFs05))
# print('The max of the SVFs computed on 0.5m is ' + str(max(SVFs05)))
# print('The min of the SVFs computed on 0.5m is ' + str(min(SVFs05)))

#
#SVFs5m = SVF5mPy.SVFs
# meanSVFs5m = sum(SVFs5m)/len(SVFs5m)
# print('The mean of the SVFs computed on 5m is ' + str(meanSVFs5m))
# print('The max of the SVFs computed on 5m is ' + str(max(SVFs5m)))
# print('The min of the SVFs computed on 5m is ' + str(min(SVFs5m)))
# print(np.sum(((np.array(SVFs5m)-meanSVFs5m)**2))/(len(SVFs5m)))


"Time elapsed"
endtime = time.time()
elapsed_time = endtime-sttime
print('Execution time:', elapsed_time, 'seconds')

