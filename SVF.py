import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import tifffile as tf
from tqdm import tqdm
#import Functions
import config
from functools import partial
import time
#import KNMI_SVF_verification
import Constants
# import Sunpos
# import SVFs05m
# import SF05mHN1
# import pickle
# import SVFs5m
# import SVFGR25
# from pynverse import inversefunc
# from scipy.optimize import curve_fit

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
            road_elements[i,j] = max(np.count_nonzero(part==0),0)
            built_elements[i,j] = max(np.count_nonzero(part>0),0)
            "The road elements are actually also water elements"
            water_elements[i,j] = max(np.count_nonzero(part==-1),0)
            delta[i,j] = built_elements[i,j]/(road_elements[i,j]+built_elements[i,j]+water_elements[i,j])
            wall_area_med[i,j] = max(np.sum(part_wall),0)

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

def average_surfacetype(matrix,data, grid_ratio):
    [x_long, y_long] = matrix.shape
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
    #data = data[:int(x_long/5),int(y_long/5):int(2*y_long/5)]
    data = data[:int(x_long/5),int(2*y_long/5):int(3*y_long/5)]
    #SVF_knmi_HN1 = SVF_knmi_HN1[:int(x_long/5),:int(y_long/5)]
    [x_len,y_len] = data.shape
#
# print(np.max(data))
# plt.figure()
# plt.imshow(data,vmin=0,vmax=50)
# plt.show()
#
# H_w = np.linspace(0.2,5,20)
# SF_w = np.empty((len(H_w)))
# SF_wall = np.empty((len(H_w)))
# SF_road = np.empty((len(H_w)))
# # Zenith = [np.pi/6,np.pi/5,np.pi/4,np.pi/3]
# # c = ['r','g','b','y']
# Azi,El = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,13,radians=True)
# Zenith = np.pi/2-El
# H_w = np.linspace(0.2,5,20)
# for h in range(len(H_w)):
#     [SF_roof, SF_wall[h], SF_road[h]] = SF_ave_masson(H_w[h],Zenith)
# popt, pcov = curve_fit(f1, SF_road, SF_wall)
# for t in range(len(Zenith)):
# #SF_w = np.empty((len(H_w)))
#     SF_wall = np.empty((len(H_w)))
#     SF_road = np.empty((len(H_w)))
#     for h in range(len(H_w)):
#         [SF_roof, SF_wall[h], SF_road[h]] = SF_ave_masson(H_w[h],Zenith[t])
#     popt, pcov = curve_fit(f1, SF_road, SF_wall)
#     "Fit versus analytic"
#     SF_road_lin = np.linspace(0.01,1,30)
#     Zen = np.around(Zenith[t]*180/np.pi)
#     print(Zen)
#     plt.plot(SF_road,SF_wall,color=c[t],linestyle='dashed', label='Masson for Zenith angle' + str(Zen))
#     plt.plot(SF_road_lin,f1(SF_road_lin,popt[0],popt[1],popt[2]),color=c[t], label='Fit for Zenith angle ' + str(Zen))
#     #plt.plot(SF_road,f1(SF_road,popt[0],popt[1],popt[2]),color=c[t], linestyle='dashed',label='Fit for Zenith angle ' + str(Zenith[t]))
#     #plt.plot(H_w,f1(SF_road,popt[0],popt[1],popt[2],popt[3]),color=c[t],label='Fit for Zenith angle ' + str(Zenith))
# plt.xlim((0,1))
# Zen = np.around(Zenith*180/np.pi)
# print(Zen)
# plt.figure()
# SF_road_lin = np.linspace(0.01,1,30)
# plt.plot(SF_road_lin,f1(SF_road_lin,popt[0],popt[1],popt[2]))
# plt.ylim((0,1))
# plt.ylabel('SF wall [0-1]')
# plt.xlabel('SF road [0-1]')
# plt.legend(loc='upper right')
# #[h_w,zenith] = np.meshgrid(H_w,Zenith)
# grid_ratio = 50
# with open('Pickles/1MaySF/SF_may1_13_HN1.pickle', 'rb') as f:
#     SF_matrix = pickle.load(f)
# [SF_ave_roof,SF_ave_road] = average_svf_surfacetype(SF_matrix,data,grid_ratio)
# SF_road_ave = SF_ave_road.flatten()
# SF_roof_ave = SF_ave_roof.flatten()
# SF_wall_ave = np.empty((len(SF_road_ave)))
# for h in range(len(SF_road_ave)):
#     SF_wall_ave[h] = f1(SF_road_ave[h], popt[0],popt[1],popt[2])
"Histograms"
# gridratio = 5
# data = average_svf(data,gridratio)
# [x_len,y_len] = data.shape
# data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#
# with open('SVFP2_Matrix.npy', 'rb') as f:
#     SVF_matrix = np.load(f)
# #grid_ratio=1
# bin_nr = 30
# SVF_roof_ave = []
# SVF_road_ave = []
# for i in range(int(x_len/2)):
#     for j in range(int(y_len/2)):
#         if data[i,j]>0:
#             SVF_roof_ave.append(SVF_matrix[i,j])
#         else:
#             SVF_road_ave.append(SVF_matrix[i,j])
# plt.figure()
# SVF_list = SVF_matrix.flatten()
# plt.hist(SVF_list, bins = bin_nr,weights=np.ones(len(SVF_list))/len(SVF_list))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF [0-1]')
# plt.figure()
# SVF_road_ave = SVF_ave_road.flatten()
# SVF_roof_ave = SVF_ave_roof.flatten()
#SVF_road_ave = np.array(SVF_road_ave)
# SVF_wall_ave = Inv_WallvsRoadMasson(np.array(SVF_road_ave))
# print(len(SVF_road_ave)/(len(SVF_road_ave)+len(SVF_roof_ave)))
# plt.figure()
# plt.hist(SVF_wall_ave, bins = bin_nr,weights=np.ones(len(SVF_wall_ave))/len(SVF_wall_ave))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF [0-1]')
# plt.figure()
# plt.hist(SVF_road_ave, bins = bin_nr,weights=np.ones(len(SVF_road_ave))/len(SVF_road_ave))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF [0-1]')
# plt.figure()
# plt.hist(SVF_roof_ave, bins = bin_nr,weights=np.ones(len(SVF_roof_ave))/len(SVF_roof_ave))
# plt.ylabel('Normalized Counts [0-1]')
# plt.xlabel('SVF [0-1]')
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
coords = coordheight(data)
blocklength = int(x_len/2*y_len/2)
SVFs = calc_SVF(coords, max_radius, blocklength, gridboxsize)
print("The SVFs non averaged")
print(SVFs)
#
gridratio = 5
data = average_svf(data,gridratio)
coords = coordheight(data)
[x_len, y_len] = data.shape
blocklength = int(x_len/2*y_len/2)
SVFs = calc_SVF(coords, max_radius, blocklength, int(gridratio*gridboxsize))
print("The SVFs averaged over 2.5m")
print(SVFs)

# SVF = SVFGR25.SVFs
#np.save('SVFP1_List', SVF)
#SVFs = calc_SVF(coords, max_radius, blocklength, int(gridboxsize*gridratio))
# SVF_matrix = np.ndarray([x_len,y_len])
# for i in range(int(x_len/2*y_len/2)):
#     SVF_matrix[int(coords[i,0]),int(coords[i,1])] = SVF[i]
# SVF_matrix = SVF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# plt.figure()
# plt.imshow(SVF_matrix, vmin=0, vmax=1)
# plt.show()
# np.save('SVF_GR25_Matrix', SVF_matrix)
#print(SVF_matrix)

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
#
# def prepare_animation(bar_container):
#
#     def animate(frame_number):
#         frame_number = (frame_number+6)
#         dataSF = getSFdata(frame_number,grid_ratio)
#         n, _ = np.histogram(dataSF, HIST_BINS)
#
#         for count, rect in zip(n, bar_container.patches):
#             rect.set_height(count)
#
#         return bar_container.patches
#
#     return animate
# fig, ax = plt.subplots()
# _, _, bar_container = ax.hist(data, HIST_BINS,lw=1,ec="blue", fc="yellow", alpha=0.5)
# #ax.set_ylim(top=1)
# ani = animation.FuncAnimation(fig, prepare_animation(bar_container), 15,repeat=True, blit=True)
# plt.show()
# #HTML(ani.to_html5_video())
# frames = [] # for storing the generated images
# fig = plt.figure()
# for i in range(15):
#     with open('pickles/1MaySF/SF_may1_'+ int(i+6) +'_HN1.pickle', 'rb') as f:
#         SF_matrix = pickle.load(f)
#     frames.append([plt.imshow(SF_matrix,animated=True)])
# ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=1000)
# writergif = animation.PillowWriter(fps=30)
# ani.save('/Users/rosaliekievits/Desktop/PlaatjesMEP/ShadowFactors_may1/Out files May 1/New/SFmovie.gif', writer=writergif)
# plt.show()
"end movie"

"Shadowfactor"
# coords = coordheight(data)
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
# coords = coordheight(data)
# hours = np.array([6, 7]) #np.linspace(13,17,5)
# SF_matrix = np.ndarray([int(x_len),int(y_len)])
# for h in range(len(hours)):
#     [azimuth,el_angle] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,hours[h],radians=True)
#     SFs = reshape_SVF(data,coords,gridboxsize,azimuth,el_angle,reshape=False,save_CSV=False,save_Im=False)
#     print("The Date is " + str(Constants.julianday) + " and time is " + str(hours[h]))
#     print(SFs)
#     for i in range(int(x_len/2*y_len/2)):
#         SF_matrix[int(coords[i,0]),int(coords[i,1])] = SF[i]
#     SF_matrix = SF_matrix[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#     print(SF_matrix)
    #np.savetxt("SFmatrix" + str(hours[h]) + ".csv", SF_matrix, delimiter=",")
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

"SVF loop for different grid ratios"
# SVF = SVFs5m.SVFs
# gridratio = 5
# data = average_svf(data,gridratio)
# [x_len,y_len] = data.shape
# #data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
# print(data.shape)
# gridratio = [1,2,5,10,25,50] #all divided by 5 already
# with open('SVF_GR5Matrix.npy', 'rb') as f:
#     SVF_matrix = np.load(f)
# mean_roof = np.ndarray((len(gridratio),1))
# min_roof = np.ndarray((len(gridratio),1))
# max_roof = np.ndarray((len(gridratio),1))
# mean_road = np.ndarray((len(gridratio),1))
# min_road = np.ndarray((len(gridratio),1))
# max_road = np.ndarray((len(gridratio),1))
# frac_roof_min = np.ndarray((len(gridratio),1))
# frac_roof_max = np.ndarray((len(gridratio),1))
# frac_roof_mean = np.ndarray((len(gridratio),1))
# frac_road_min = np.ndarray((len(gridratio),1))
# frac_road_max = np.ndarray((len(gridratio),1))
# frac_road_mean = np.ndarray((len(gridratio),1))
#
# for i in range(len(gridratio)):
#     Roof_frac, Wall_frac, Road_frac = geometricProperties(data,gridratio[i],gridboxsize)
#     [SVF_roof,SVF_road] = average_svf_surfacetype(SVF_matrix,data,gridratio[i])
#     #data = data[int(x_len/4):int(3*x_len/4),int(y_len/4):int(3*y_len/4)]
#     # SVF_wall = 1-average_svf(SVF_matrix,gridratio[i])
#     mean_roof[i] = np.nanmean(SVF_roof)#*np.mean(Roof_frac)
#     max_roof[i] = np.nanmax(SVF_roof)
#     min_roof[i] = np.nanmin(SVF_roof)
#     frac_roof_mean[i] = np.nanmean(Roof_frac)
#     frac_roof_min[i] = np.nanmin(Roof_frac)
#     frac_roof_max[i] = np.nanmax(Roof_frac)
#     mean_road[i] = np.nanmean(SVF_road)#*np.mean(Roof_frac)
#     max_road[i] = np.nanmax(SVF_road)
#     min_road[i] = np.nanmin(SVF_road)
#     frac_road_mean[i] = np.nanmean(Road_frac)
#     frac_road_min[i] = np.nanmin(Road_frac)
#     frac_road_max[i] = np.nanmax(Road_frac)
#
# plt.figure()
# plt.plot(gridratio, mean_roof,'r',label='Mean SVF')
# plt.plot(gridratio, min_roof,'r',linestyle='dashed',label='Min SVF')
# plt.plot(gridratio, max_roof,'r',linestyle='dotted',label='Max SVF')
# plt.plot(gridratio, frac_roof_mean,'b',label='Mean Area Fraction')
# plt.plot(gridratio, frac_roof_min,'b',linestyle='dashed',label='Min Area Fraction')
# plt.plot(gridratio, frac_roof_max,'b',linestyle='dotted',label='Max Area Fraction')
# plt.xlabel('gridratio')
# plt.legend(loc='upper right')
# plt.ylabel('SVF, Area Fraction [0-1]')
# "Road"
# plt.figure()
# plt.plot(gridratio, mean_road,'r',label='Mean SVF')
# plt.plot(gridratio, min_road,'r',linestyle='dashed',label='Min SVF')
# plt.plot(gridratio, max_road,'r',linestyle='dotted',label='Max SVF')
# plt.plot(gridratio, frac_road_mean,'b',label='Mean Area Fraction')
# plt.plot(gridratio, frac_road_min,'b',linestyle='dashed',label='Min Area Fraction')
# plt.plot(gridratio, frac_road_max,'b',linestyle='dotted',label='Max Area Fraction')
# plt.xlabel('gridratio')
# plt.legend(loc='upper right')
# plt.ylabel('SVF, Area Fraction [0-1]')
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
# [SF_roof,SF_road] = average_svf_surfacetype(SF_matrix,data,gridratio)
# Roof_frac, Wall_frac, Road_frac = geometricProperties(data,gridratio,gridboxsize)
# SF_wall = (1-average_svf(SF_matrix,gridratio))/Wall_frac
# # SF_wall[Wall_frac<eps] = 0
# # SF_road[Road_frac==0] = 0
# # SF_roof[Roof_frac==0] = 0
# print(np.min(SF_wall))
# print(np.max(SF_wall))
# print(np.mean(SF_wall))


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

"Compute the Wall and Sky view factor with the algorithm for an infinite canyon for all angles"
# timesteps = 24*6
# Azi = np.empty((timesteps))
# El = np.empty((timesteps))
# for t in range(timesteps):
#     hour = (t+1)*(Constants.timestep/3600)
#     Azi[t],El[t] = Sunpos.solarpos(Constants.julianday,Constants.latitude,Constants.Long_lagos,hour,radians=True)
# height_l = 30
# height_r = 30
# width = int(50/gridboxsize)
# h_w = height_l/(width*gridboxsize)
# len_mat = 500 # shape of the matrix
# gridratio = 500
# ucm_matrix = np.ones((len_mat,len_mat))*height_l
# ucm_matrix[:,int(len_mat/2-width/2):int(len_mat/2+width/2)] = 0
# ucm_matrix[:,int(len_mat/2+width/2)+1::] = height_r
# coords = coordheight(ucm_matrix)
# SF_r = np.empty((timesteps))
# SF_w = np.empty((timesteps))
# SF_r_2 = np.empty((timesteps))
# [Roof_frac, Wall_frac, Road_frac] = geometricProperties(ucm_matrix,gridratio,gridboxsize)
# for t in range(timesteps):
#     SF_canyon = np.empty((width))
#     SF_canyon_2 = np.empty((width))
#     if El[t]>0:
#         SF_matrix = np.ones((ucm_matrix.shape))
#         for j in range(width):
#             point_canyon = [len_mat/2,len_mat/2-width/2+j,0]
#             SF_canyon[j] = shadowfactor(point_canyon,coords,Azi[t],El[t])
#             #SF_canyon_2[j] = shadowfactor(point_canyon,coords,np.pi/2,El[t])
#             SF_matrix[:,int(len_mat/2-width/2+j)] = SF_canyon[j]
#         SF_r[t] = np.mean(SF_canyon)
#         #SF_r_2[t] = np.mean(SF_canyon_2)
#         SF_w[t] = (1-np.mean(SF_matrix))/(2*Wall_frac)
#     else:
#         SF_r[t] = 0
#         SF_w[t] = 0
#


# with open('SF_wall_24_Hours_may1.npy', 'rb') as f:
#     SF_w = np.load(f)
# with open('SF_road_24_Hours_may1.npy', 'rb') as f:
#     SF_r = np.load(f)

# Zenith = np.pi/2-El
# def SF_masson(h_w,Zenith):
#     lamb_zero = np.arctan(1/h_w)
#     SF_roof = 1
#     if (Zenith > lamb_zero):
#         SF_wall = (1/2/h_w)
#         SF_road = 0
#     elif (Zenith < lamb_zero):
#         SF_wall = (1/2*np.tan(Zenith))
#         SF_road = 1-h_w*np.tan(Zenith)
#     if Zenith>np.pi/2:
#         SF_wall = 0
#         SF_road = 0
#         SF_roof = 0
#     return SF_roof, SF_wall, SF_road
#
# SF_roof_m = np.empty((timesteps))
# SF_road_m = np.empty((timesteps))
# SF_wall_m = np.empty((timesteps))
#
# for t in range(timesteps):
#     [SF_roof_m[t], SF_wall_m[t], SF_road_m[t]] = SF_masson(h_w,Zenith[t])
#
# with open('pickles/SF_wall_24_Hours_may1', 'wb') as f:
#     pickle.dump(SF_w, f)
# with open('pickles/SF_road_24_Hours_may1', 'wb') as f:
#     pickle.dump(SF_r, f)
# with open('pickles/SF_road_24_Hours_may1', 'rb') as f:
#     SF_r = pickle.load(f)
# with open('pickles/SF_wall_24_Hours_may1', 'rb') as f:
#     SF_w = pickle.load(f)
# print(SF_r.shape)
# print(SF_w.shape)
# plt.figure()
# time_lin = np.linspace(0,timesteps,timesteps)*Constants.timestep/3600
# plt.plot(time_lin,SF_r, label="road")
# # plt.plot(time,SF_r_2, label="road 2")
# # # plt.plot(time,SF_road_m, label="road masson")
# # # plt.plot(time,SF_wall_m, label="wall masson")
# plt.xlabel('Time [h]')
# plt.ylabel('Shadowfactor [0-1]')
# plt.plot(time_lin,SF_w, label="wall")
# plt.legend()
# plt.figure()
# plt.ylabel('Angle [Rad]')
# plt.xlabel('Time [h]')
# plt.plot(time,El, label="El")
# plt.plot(time,Azi, label="Azimuth")
# plt.legend()
# plt.show()

"Compare the Sky view factor of masson for an urban geometry"
# widths = np.linspace(10,15,6)/gridboxsize
# SFs = np.ndarray((len(widths),1))
# SF_w = np.ndarray((len(widths),1))
# H_w = np.ndarray((len(widths),1))
# sf_masson_roads = np.ndarray((len(widths),1))
# phi_masson_roads = np.ndarray((len(widths),1))
# sf_masson_walls= np.ndarray((len(widths),1))
# height_r = 15
# height_l = 15
# len_mat = 500 # shape of the matrix
# plt.figure()
# gridratio = 500
# phi_masson_walls = np.ndarray((len(widths),1))
# # SVFs = np.ndarray((len(widths),1))
# # SVF_roofs = np.ndarray((len(widths),1))
# # SVFs_w = np.ndarray((len(widths),1))
# # SFs_w_alg = np.ndarray((len(widths),1))
# # SFs_w_alg_2 = np.ndarray((len(widths),1))
# # SVFs_w_2 = np.ndarray((len(widths),1))
# Roof_frac = np.ndarray((len(widths),1))
# Wall_frac = np.ndarray((len(widths),1))
# Road_frac = np.ndarray((len(widths),1))
# RoadVF_wall = np.ndarray((len(widths),1))
# RoadVF_wall_alg = np.ndarray((len(widths),1))
# el_angles = [5*np.pi/12,5*np.pi/15,5*np.pi/18,5*np.pi/21]
# colours = ['r','b','y','g']
# for e in range(len(el_angles)):
#     elevation_angle = el_angles[e]
#     Zenith = np.pi/2-elevation_angle
#     # H_ws = np.linspace(0.2,5,20)
#     # SF_wall = np.empty((len(H_ws)))
#     # SF_road = np.empty((len(H_ws)))
#     # for h in range(len(H_ws)):
#     #     [SF_roof, SF_wall[h], SF_road[h]] = SF_ave_masson(H_ws[h],Zenith)
#     # popt, pcov = curve_fit(f1, SF_road, SF_wall)
#     for i in range(len(widths)):
#         int(widths[i])
#         h_w = height_l/(widths[i]*gridboxsize)
#         zenith = np.pi/2-elevation_angle
#         lamb_zero = np.arctan(1/h_w)
#         H_w[i] = h_w
#         ucm_matrix = np.ones((len_mat,len_mat))*height_l
#         ucm_matrix[:,int(len_mat/2-widths[i]/2):int(len_mat/2+widths[i]/2)] = 0
#         ucm_matrix[:,int(len_mat/2+widths[i]/2)+1::] = height_r
#         coords = coordheight(ucm_matrix)
#         [Roof_frac[i], Wall_frac[i], Road_frac[i]] = geometricProperties(ucm_matrix,gridratio,gridboxsize)
#         SF_matrix = np.ones((ucm_matrix.shape))
#         phi_masson_roads[i] = np.sqrt((h_w**2+1))-h_w
#         phi_masson_walls[i] = 1/2*(h_w+1-np.sqrt(h_w**2+1))/h_w
#         SF_canyon = np.ndarray((int(widths[i]),1))
#         SF_roof = np.ndarray((int(widths[i]),1))
#         #SVF_w = np.ndarray((int(widths[i]),1))
#         SVF = np.ndarray((int(widths[i]),1))
#         if (zenith > lamb_zero):
#             sf_masson_walls[i] = 1/2/h_w
#             sf_masson_roads[i] = 0
#         elif (zenith < lamb_zero):
#             sf_masson_walls[i] = 1/2*np.tan(zenith)
#             sf_masson_roads[i] = 1-h_w*np.tan(zenith)
#         for j in range(int(widths[i])):
#             point_canyon = [len_mat/2,len_mat/2-widths[i]/2+j,0]
#             point_roof = [len_mat/2,len_mat/2-widths[i]/2-1-j,height_l]
#             SF_canyon[j] = shadowfactor(point_canyon,coords,np.pi/2,elevation_angle)
#             SF_roof[j] = shadowfactor(point_roof,coords,np.pi/2,elevation_angle)
#             #print(SF_roof)
#             SF_matrix[:,int(len_mat/2-widths[i]/2+j)] = SF_canyon[j]
#             SF_matrix[:,int(len_mat/2-widths[i]/2-1-j)] = SF_roof[j]
#             # SVF[j] = SkyViewFactor(point_canyon,coords,max_radius,gridboxsize)
#         #print("The wall fraction is " + str(np.mean(Wall_frac[i])))
#         SFs[i] = np.mean(SF_canyon)
#         # [SF_ave_roof,SF_ave_road] = average_svf_surfacetype(SF_matrix,ucm_matrix,gridratio)
#         #SF_wall_fit = f1(SFs[i],popt[0],popt[1],popt[2])
#         SF_w[i] = (1-np.mean(SF_matrix))/(2*Wall_frac[i])
#         # point_roof_l = [len_mat/2,len_mat/2-widths[i]/2-1,height_l]
#         # p_wall_2 = [len_mat/2,len_mat/2+widths[i]/2,height_r]
#         # p_wall = [len_mat/2,len_mat/2-widths[i]/2-1,height_l]
#         # SVF_roofs[i] = SkyViewFactor(point_roof_l,coords,max_radius,gridboxsize)
#         # SVFs_w[i] = SVF_wall(p_wall,coords,max_radius,1,height_l)
#         # SVFs_w_2[i] = SVF_wall(p_wall_2,coords,max_radius,3,height_r)
#
#         # SFs_w_alg[i] =  SF_wall(p_wall,coords,1,height_l,np.pi/2,elevation_angle)
#         # SFs_w_alg_2[i] = SF_wall(p_wall_2,coords,3,height_r,np.pi/2,elevation_angle)
#         #print(SF_roof)
#         # SVFs[i] = np.mean(SVF)
#         # WVF_road = (1-SVFs[i])/2
#         # RoadVF_wall[i] = phi_masson_walls[i] #SVF_w
#         # RoadVF_wall_alg[i] = (WVF_road/2 * Road_frac[i])/Wall_frac[i]
#     plt.plot(H_w,SF_w,colours[e],label='Numerical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
#     plt.plot(H_w,sf_masson_walls,colours[e],linestyle='dotted',label='Analytical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
#     #plt.plot(H_w,f1(SFs,popt[0],popt[1],popt[2]))
#     # print(SF_w)
#     # print(sf_masson_walls)
# # print(SFs_w_alg)
# # print(SFs_w_alg_2)
# #print(phi_masson_walls)
# # plt.plot(H_w,phi_masson_walls,'blue',linestyle='dotted',label='Analytical, wall')
# # plt.plot(H_w,SVFs_w,'blue',label='Alg, left wall')
# # plt.plot(H_w,SVFs_w_2,'lightblue',label='Alg, right wall')
# # plt.plot(H_w,SVF_roofs,'red',label='Roof')
# # plt.plot(H_w,SVFs,'y',label='Road')
# # plt.plot(H_w,RoadVF_wall,'r',linestyle='dotted',label='Road VF wall')
# # plt.plot(H_w,RoadVF_wall_alg,'r',label='Road VF, wall, alg')
#
# #plt.plot(H_w,phi_masson_roads,'y',linestyle='dotted',label='Analytical, road')
# plt.xlabel('Width ratio')
# plt.ylabel('SF [0-1]')
# plt.legend(loc='upper right')
# plt.ylim((0,1))
# plt.figure()
# plt.plot(H_w,Wall_frac,'r',linestyle='dotted',label='Wall_frac')
# plt.plot(H_w,Road_frac,'b',linestyle='dotted',label='Road frac')
# plt.plot(H_w,Roof_frac,'y',linestyle='dotted',label='Roof frac')
# plt.ylim((0,1))
# #plt.plot(widths,SVFs,'red',label='Analytical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
# #plt.plot(widths,SVFs,'red',label='Analytical, elevation angle = ' + str(np.round(elevation_angle*180/np.pi)) + 'degrees')
#
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

"Infinite canyon Time evolution"
# h_w = 0.6
# SVF_road = (np.sqrt((h_w**2+1))-h_w)*np.ones([2,2])
# SVF_wall = (1/2*(h_w+1-np.sqrt(h_w**2+1))/h_w)*np.ones([2,2])
# SVF_roof = np.ones([2,2])

"Time elapsed"
endtime = time.time()
elapsed_time = endtime-sttime
print('Execution time:', elapsed_time, 'seconds')

