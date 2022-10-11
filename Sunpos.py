import numpy as np
import PyAstronomy as PA
latitude = 52
julianday = 22284 #11 oct
hour = 16



def solarpos(julian_day,latitude,hour):
    """
    :param julian_day: day of the year (1 is jan1, 365 is dec31)
    :param latitude: latitude of location
    :param hour: hour of the day, can be fractional
    :return: elevation angle and azimuth of the sun
    """
    """Correct the hour to GMT (summertime in Rotterdam)"""
    hour = hour-2
    latitude = latitude*np.pi/180
    day_rel = (julian_day*np.pi/180)/365.25
    delta = np.arcsin(0.3987*np.sin(day_rel-1.4+0.0355*np.sin(day_rel-0.0489)))

    omega_sunset = np.arccos(-np.tan(latitude)*np.tan(delta))
    omega_sunrise = -omega_sunset
    """the sun change 15degrees or 0.26 radians per hour"""
    hour_rad = (hour-12)*0.261799
    """the declination angle"""
    alpha = np.arcsin(np.sin(delta)*np.sin(latitude)+np.cos(delta)*np.cos(latitude)*np.cos(hour_rad))

    if (hour_rad>omega_sunset or hour_rad<omega_sunrise):
        alpha = 0
    sin_azimuth = (np.sin(latitude)*np.sin(alpha)-np.sin(delta))/(np.cos(latitude)*np.cos(alpha))
    cos_azimuth = np.cos(delta)*np.sin(hour_rad)/np.cos(alpha)
    if sin_azimuth>0:
        azimuth = np.arccos(cos_azimuth)
    elif sin_azimuth<0:
        azimuth = -np.arccos(cos_azimuth)
    return azimuth,alpha

PA.pyasl.sunpos(julianday, )
