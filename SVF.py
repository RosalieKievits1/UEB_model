import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
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
steps_beta = 360 # so we range in steps of 1 degree
max_radius = 500
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
    "this is the analytical dome area"
    dome_area = max_radius**2*2*np.pi
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
        # beta_min = - np.arcsin(np.sqrt(2*gridboxsize**2)/2/dome_p[d,3]) + dome_p[d,4]
        # beta_max = np.arcsin(np.sqrt(2*gridboxsize**2)/2/dome_p[d,3]) + dome_p[d,4]

        """Where the index of betas fall within the min and max beta, and there is not already a larger psi blocking"""
        betas[np.nonzero(np.logical_and((betas < psi), np.logical_and((beta_min <= betas_lin), (betas_lin < beta_max))))] = psi
        d += 1
    areas = d_area(betas, steps_beta, max_radius)
    """The SVF is the fraction of area of the dome that is not blocked"""
    SVF = np.around((dome_area - np.sum(areas))/dome_area, 3)
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
    beta_min = np.asarray(- np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)
    beta_max = np.asarray(np.arcsin(np.sqrt(2*gridboxsize**2)/2/radii) + azimuth)
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
    #blocklength = int(x_len/2*y_len/2)
    blocklength = int((x_len-2*max_radius/gridboxsize)*(y_len-2*max_radius/gridboxsize))
    "Compute SVF and SF and Reshape the shadow factors and SVF back to nd array"
    SVFs = calc_SVF(coords, max_radius, blocklength, gridboxsize)
    #SFs = calc_SF(coords,azimuth,zenith,blocklength)
    "If reshape is true we reshape the arrays to the original data matrix"
    if (reshape == True) & (SVFs is not None):
        SVF_matrix = np.ndarray([x_len-2*max_radius/gridboxsize,y_len-2*max_radius/gridboxsize])
        #SVF_matrix = np.ndarray([x_len/2,y_len/2])
        #SF_matrix = np.ndarray([x_len,y_len])
        for i in range(blocklength):
            SVF_matrix[coords[i,0]-max_radius/gridboxsize,coords[i,1]-max_radius/gridboxsize] = SVFs[i]
            #SVF_matrix[coords[i,0]-x_len/2,coords[i,1]-y_len/2] = SVFs[i]
            #SF_matrix[coords[i,0]-x_len/2,coords[i,1]-y_len/2] = SFs[i]
        if save_CSV == True:
            np.savetxt("SVFmatrix.csv", SVF_matrix, delimiter=",")
            #np.savetxt("SFmatrix.csv", SF_matrix, delimiter=",")
        if save_Im == True:
            tf.imwrite('SVF_matrix.tif', SVF_matrix, photometric='minisblack')
            #tf.imwrite('SF_matrix.tif', SF_matrix, photometric='minisblack')
        return SVF_matrix#,SF_matrix
    #
    elif (reshape == False) & (SVFs is not None):
        if save_CSV == True:
            np.savetxt("SVFs" + str(gridboxsize) + ".csv", SVFs, delimiter=",")
            #np.savetxt("SFs.csv", SFs, delimiter=",")
        return SVFs#, SFs


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
    [Wall_area, wall_area_total] = wallArea(data,gridboxsize)

    Total_area = Roof_area + wall_area_total + Road_area + Water_area
    """Fractions of the area of the total surface"""
    Roof_frac = np.around(Roof_area/Total_area,3)
    Wall_frac = np.around(wall_area_total/Total_area,3)
    Road_frac = np.around(Road_area/Total_area,3)
    Water_frac = np.around(Water_area/Total_area,3)

    H_W = ave_height*delta
    return ave_height, delta, Roof_area, wall_area_total, Road_area,Water_area, Roof_frac, Wall_frac, Road_frac, Water_frac, H_W

def average_svf(SVF_matrix, grid_ratio):
    [x_long, y_long] = SVF_matrix.shape
    SVF_ave = np.ndarray([int(x_long/grid_ratio),int(y_long/grid_ratio)])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(x_long):
        for j in range(y_long):
            part = SVF_matrix[i*grid_ratio:(i+1)*grid_ratio, j*grid_ratio:(j+1)*grid_ratio]
            SVF_ave[i,j] = np.mean(part)
    return SVF_ave

def wallArea(data,gridboxsize):
    """
    :param data: Dataset to compute the wall area over
    :param gridboxsize: size of the grid cells
    :return: the wallarea matrix and total wall area
    """
    """Matrix of ones where there are buildings"""
    [x_len,y_len] = [data.shape[0], data.shape[1]]
    """Set all the water elements to 0 height again"""
    data[data<0] = 0
    """We only evaluate the area in the center block"""
    wall_area = np.ndarray([int(x_len/2),int(y_len/2)])
    if (gridboxsize == 0.5):
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
                    wall_area[int(i-x_len/4),int(j-x_len/4)] = 0
                i+=1
                j+=1
    elif (gridboxsize==5):
        i = int(max_radius/gridboxsize)
        j = int(max_radius/gridboxsize)
        while i < int(x_len-max_radius/gridboxsize):
            while j < int(y_len-max_radius/gridboxsize):
                if (data[i,j]>0):
                    """We check for all the points surrounding the building if they are also buildings, 
                    if the building next to it is higher the wall belongs to the building next to it,
                    if the current building is higher, the exterior wall is the difference in height * gridboxsize"""
                    wall1 = max(data[i,j]-data[i+1,j],0)*gridboxsize
                    wall2 = max(data[i,j]-data[i-1,j],0)*gridboxsize
                    wall3 = max(data[i,j]-data[i,j+1],0)*gridboxsize
                    wall4 = max(data[i,j]-data[i,j-1],0)*gridboxsize
                    """The wall area corresponding to that building is"""
                    wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize)] = wall1+wall2+wall3+wall4
                elif (data[i,j]==0):
                    wall_area[int(i-max_radius/gridboxsize),int(j-max_radius/gridboxsize)] = 0
                i+=1
                j+=1
    """wall_area is a matrix of the size of center block of data, 
    with each point storing the the exterior wall for that building,
    wall_area_total is the total exterior wall area of the dataset"""
    wall_area_total = np.sum(wall_area)
    return wall_area, wall_area_total

"The block is divided into 25 blocks, this is still oke with the max radius but it does not take to much memory"

# steps_beta_lin = np.linspace(90,720,8)
# SVFs = np.zeros(len(steps_beta_lin))
# point = coords[int(blocklength),:]
# for i in range(len(steps_beta_lin)):
#     SVFs[i] = SkyViewFactor(point,coords,max_radius,gridboxsize,int(steps_beta_lin[i]))

#print(SVFs)
# plt.figure()
# plt.plot(steps_beta_lin,SVFs)
# plt.xlabel("Angular steps")
# plt.ylabel("SVF")
# plt.show()
"Here we print the info of the run:"
print("gridboxsize is " + str(gridboxsize))
print("max radius is " + str(max_radius))
print("part is 1st up, 1st left")
print("Data block is HN1")
#
"Switch for 0.5 or 5 m"
download_directory = config.input_dir_knmi
SVF_knmi_HN1 = "".join([download_directory, '/SVF_r37hn1.tif'])
SVF_knmi_HN1 = tf.imread(SVF_knmi_HN1)
SVF_knmi_HN1[SVF_knmi_HN1>1] = 0
SVF_knmi_HN1[SVF_knmi_HN1<0] = 0
print('SVF_knmi is read')
grid_ratio = int(gridboxsize/gridboxsize_knmi)
if gridboxsize==5:
    dtm_HN1 = "".join([input_dir, '/M5_37HN1.TIF'])
    dsm_HN1 = "".join([input_dir, '/R5_37HN1.TIF'])
    data = readdata(minheight,dsm_HN1,dtm_HN1)
    [x_long, y_long] = data.shape
    SVF_means = np.ndarray([x_long,y_long])
    "We want to take the mean of the SVF values over a gridsize of gridratio"
    for i in range(x_long):
        for j in range(y_long):
            part = SVF_knmi_HN1[i*grid_ratio:(i+1)*grid_ratio, j*grid_ratio:(j+1)*grid_ratio]
            SVF_means[i,j] = np.mean(part)
    SVF_knmi_HN_1 = SVF_means
elif gridboxsize==0.5:
    dtm_HN1 = "".join([input_dir, '/M_37HN1.TIF'])
    dsm_HN1 = "".join([input_dir, '/R_37HN1.TIF'])
    data = readdata(minheight,dsm_HN1,dtm_HN1)
    [x_long, y_long] = data.shape
    data = data[:int(x_long/5),:int(y_long/5)]
    SVF_knmi_HN1 = SVF_knmi_HN1[:int(x_long/5),:int(y_long/5)]
coords = coordheight(data,gridboxsize)
print('coords array is made')
SVFs = reshape_SVF(data, coords,gridboxsize,300,20,reshape=False,save_CSV=True,save_Im=False)
print(SVFs)
KNMI_SVF_verification.Verification(SVFs,SVF_knmi_HN1,gridboxsize,max_radius,gridboxsize_knmi,matrix=False)

"Fisheye plot"
# # linksboven
# dtm1 = "".join([input_dir, '/M_37HN1.TIF'])
# dsm1 = "".join([input_dir, '/R_37HN1.TIF'])
# data = readdata(minheight,dsm1,dtm1)
# [x_len,y_len] = data.shape
# coords = coordheight(data,gridboxsize)
# "Make fisheye plot"
#
# blocklength = x_len/2*y_len/2
# point = coords[int(blocklength),:]
# bottom = 0
# max_area = max_radius**2 * 2 * np.pi / steps_beta
# [svf, areas] = SkyViewFactor(point,coords,max_radius,gridboxsize)
# print(svf)
# #print(areas)
# theta = np.linspace(0.0, 2 * np.pi, steps_beta, endpoint=False)
# radii = - areas + max_area
# width = (2*np.pi) / steps_beta
#
# ax = plt.subplot(111, polar=True)
# bars = ax.bar(theta, radii, width=width, bottom=bottom)
# ax.set_facecolor("grey")
# ax.get_yaxis().set_ticks([])
# ax.get_yaxis().set_visible(False)
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
#
# # Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor("lightblue")#(plt.cm.jet(r / 10.))
#     bar.set_alpha(0.8)
#
# plt.show()

"Time elapsed"
endtime = time.time()
elapsed_time = endtime-sttime
print('Execution time:', elapsed_time, 'seconds')
