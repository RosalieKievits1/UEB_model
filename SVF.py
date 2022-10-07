import numpy as np
"""Now we want to calculate the sky view factor"""
steps_theta = 90 # so we range in steps of 4 degrees
steps_phi = 90 # so we range in steps of 2 degrees
max_radius = 30 # max radius is 15 m
   # horizontal angle over which we move
#phi = np.linspace(0,np.pi,steps_phi)        # vertical angle over which we move

def solarpos(julian_day,latitude,hour):
    """
    :param julian_day: day of the year (1 is jan1, 365 is dec31)
    :param latitude: latitude of location
    :param hour: hour of the day, can be fractional
    :return: elevation angle and azimuth of the sun
    """
    day_rel = (julian_day*2*np.pi)/365.25
    delta = np.arcsin(0.3987*np.sin(day_rel-1.4+0.0355*np.sin(day_rel-0.0489)))
    omega_sunset = np.arccos(-np.tan(latitude)*np.tan(delta))
    omega_sunrise = -omega_sunset
    # the sun change 15degrees or 0.26 radians per hour
    hour_rad = hour*0.261799
    # the declination angle
    alpha = np.arcsin(np.sin(delta)*np.sin(latitude)+np.cos(delta)*np.cos(latitude)*np.cos(hour_rad))
    if (hour_rad>omega_sunset or hour_rad<omega_sunrise):
        alpha = 0
    sin_azimuth = (np.sin(latitude)*np.sin(alpha)-np.sin(delta))/(np.sin(latitude)*np.sin(alpha))
    cos_azimuth = np.cos(delta)*np.sin(hour_rad)/np.cos(alpha)
    if sin_azimuth>0:
        azimuth = np.arccos(cos_azimuth)
    elif sin_azimuth<0:
        azimuth = -np.arccos(cos_azimuth)
    return azimuth,alpha


"""First we store the data in a more workable form"""
def coordheight(data):
    """
    create an array with the length for each tile, and the 3 columns for x, y, and z
    :param data: the data array with the height for each tile
    :return: 3 columns for x, y, and z
    """
    coords = np.ndarray([data.shape[0]*data.shape[1],3])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            coords[i+j,0] = i
            coords[i+j,1] = j
            coords[i+j,2] = data[i,j]
    return coords

def dist(point, coord):
    """
    :param point: evaluation point (x,y,z)
    :param coord: array of coordinates with heights
    :return: the distance from each coordinate to the point and the angle
    """
    dx = (point[0] - coord[0])
    dy = (point[1] - coord[1])
    dist = np.sqrt(abs(dx)**2 + abs(dy)**2)
    angle = np.arctan(dy/dx)
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
        radius[i],angle[i] = dist(point,coords[i])
    coords = np.concatenate([coords,radius],axis=1)
    coords = np.concatenate([coords,angle],axis=1)
    # the dome consist of points higher than the view height and within the radius we want
    dome = coords[np.logical_and(coords[:,3]<maxR,coords[:,3]>0.1)]
    dome = dome[dome[:,2]>point[2]]
    return dome

def d_area(radius, d_theta,height,maxR):
    # we assume the dome is large, and d_theta and d_phi small enough,
    # such that we can approximate the d_area as a flat surface
    dx = radius*2*np.sin(d_theta/2)
    #dz = radius*2*np.sin(d_phi/2)
    # multiply by maxR over radius such that the small area gets projected on the dome
    d_area = (dx*height)*maxR/radius
    return d_area

def calc_SVF(coords, steps_phi , steps_theta,max_radius):
    phi_lin = np.linspace(0,np.pi,steps_phi)        # vertical angle over which we move
    thetas = np.zeros(steps_theta)
    #theta_lin = np.linspace(0,2*np.pi,steps_theta)
    d_theta = np.pi/steps_theta
    d_phi = np.pi/steps_phi
    SVF = np.ndarray([coords.shape[0],1])
    for i in range(coords.shape[0]):
        # the point we are currently evaluating
        point = coords[i,:]
        # we throw away all point outside the dome
        # dome is now a 5 column array of points:
        # the 5 columns: x,y,z,radius,angle theta
        dome_p = dome(point, coords, max_radius)
        # we loop over all points inside the dome
        for d in range(dome_p.shape[0]):
            # now comes the difficult part:
            # we want to know if when we evaluate one point
            for p in range(phi_lin.shape[0]):
                # if the height of the point is higher than the height of the point
                h_phi = np.tan(phi_lin[p])*dome_p[d,3] #*max_radius #dome_p[d,2]
                if (dome_p[d,2]>=h_phi):
                    # if there already exist a blocking for that angle theta:
                    # calculate of the area of this blocking is more
                    heightdif = dome_p[d,2]-point[2]
                    area = d_area(dome_p[d,3],d_theta,heightdif,max_radius)
                    round_angle_index = int(np.floor((dome_p[d,4]*steps_theta)/(2*np.pi)))
                    if (thetas[round_angle_index]<area):
                        thetas[round_angle_index] = area
        #this is the analytical dome area but should make same assumption as for d_area
        # dome_area = max_radius**2*2*np.pi
        dome_area = (max_radius*np.sin(d_phi/2)*2)*(max_radius*np.sin(d_theta/2)*2)*steps_theta*steps_phi
        #print(sum(thetas,0))
        SVF[i] = (dome_area - sum(thetas,0))/dome_area
    return SVF

def shadowfactor(coords, azimuth,elevation_angle):
    """
    :param coords: all other points, x,y,z values
    :param azimuth: azimuth of the sun
    :param elevation_angle: elevation angle of the sun
    :return: the shadowfactor of that point:
    """
    for i in range(coords.shape[0]):
        # the point we are currently evaluating
        point = coords[i,:]
        angle = np.ndarray([coords.shape[0],1])
        radius = np.ndarray([coords.shape[0],1])
        for i in range(coords.shape[0]):
            radius[i],angle[i] = dist(point,coords[i])
        coords = np.concatenate([coords,radius],axis=1)
        coords = np.concatenate([coords,angle],axis=1)
        # coords is now a 5 column array of points:
        # the 5 columns: x,y,z,radius,angle theta with point p
    # now we want to delete all coords that have a different angle with the sun
    # next we want to check of point that are in the same azimuth of the sun are higher,
    # Set the shadowfactor to 0 or 1
