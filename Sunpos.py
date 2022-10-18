import numpy as np
"""Longitud and latitude of rotterdam in degrees"""


def solarpos(julian_day,latitude,longitude,hour,radians=True):
    """
    :param julian_day: day of the year (1 is jan1, 365 is dec31)
    :param latitude: latitude of location
    :param hour: hour of the day, can be fractional
    :return: zenith (elevation angle) and azimuth of the sun
    """

    """correct time"""
    """according to longitude of the timezone of the netherlands in degrees"""
    long_tz = 15
    day_rel = (julian_day)*360/365.25
    EOT = -0.128*np.sin(day_rel-2.80)-0.165*np.sin(2*day_rel+19.7)
    T = hour-1+(longitude-long_tz)/15+EOT

    """Compute the daily declination value"""
    latitude = latitude*np.pi/180
    day_rel= day_rel*np.pi/180
    delta = np.arcsin(0.3987*np.sin(day_rel-1.4+0.0355*np.sin(day_rel-0.0489)))

    omega_sunset = np.arccos(-np.tan(latitude)*np.tan(delta))
    omega_sunrise = -omega_sunset
    """the sun change 15 degrees or 0.26 radians per hour"""
    hour_rad = (T-12)*0.261799
    """the declination angle (acitually the zenith is defined as angle from top so not the same)"""
    zenith = np.arcsin(np.sin(delta)*np.sin(latitude)+np.cos(delta)*np.cos(latitude)*np.cos(hour_rad))

    azimuth = np.arcsin(np.cos(delta)*np.sin(hour_rad)/np.cos(zenith))
    if (np.cos(hour_rad)<(np.tan(delta)/np.tan(latitude))):
        if T<12:
            azimuth = abs(azimuth)+np.pi
        elif T>12:
            azimuth = np.pi-azimuth

    """Transform to degrees if we set the boolean radians to false"""
    """The azimuth angle is 0 when the sun is south and positive westwards"""
    if (radians==False):
        azimuth = azimuth*180/np.pi
        azimuth = azimuth+180
        zenith = zenith*180/np.pi

    """Correct to measure from north"""
    azimuth = azimuth+np.pi

    return azimuth,zenith

