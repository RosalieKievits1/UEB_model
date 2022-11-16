import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import csv
import Constants

"""Longitude and latitude of rotterdam in degrees"""


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
    T = hour+(longitude-long_tz)/15+EOT

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

    """Correct to measure from north"""
    azimuth = azimuth+np.pi

    """Transform to degrees if we set the boolean radians to false"""
    """The azimuth angle is 0 when the sun is south and positive westwards"""
    if (radians==False):
        azimuth = azimuth*180/np.pi
        azimuth = azimuth % 360
        zenith = zenith*180/np.pi

    return azimuth,zenith,omega_sunrise,omega_sunset


# df = pd.read_csv('SolarposNov1.csv')
# print(df)
#
# with open("SolarposNov1.csv", 'r') as file:
#     a = []
#     gamma = []
#     t = []
#     csvreader = csv.reader(file, delimiter=';')
#     for row in csvreader:
#         t.append(row[0])
#         a.append(row[1])
#         gamma.append(row[2])
#
# t = t[1::4]
# a = a[1::4]
# gamma = gamma[1::4]
# for i in range(len(a)):
#     a[i] = float(a[i])
#     gamma[i] = float(gamma[i])
#

# hour = np.linspace(0,24,25, dtype=int)
# Julianday = np.linspace(305,311,7,dtype=int)
# plt.figure()
# azis = np.zeros([len(hour)])
# zens = np.zeros([len(hour)])
# for d in range(len(Julianday)):
#     for t in range(len(hour)):
#         [azis[t], zens[t]] = solarpos(Julianday[d],Constants.latitude,Constants.long_rd,hour[t],radians=False)
#     plt.plot(hour,zens,label=str(Julianday[d]-304) + ' Nov')
# plt.xlabel('time [h]')
# plt.legend(loc='upper left')
# plt.ylabel('angle [degrees]')
# plt.show()

# hour = np.linspace(10,14,24)
# Julianday = np.linspace(305,312,8,dtype=int)
# plt.figure()
# azis = np.zeros([len(hour)])
# zens = np.zeros([len(hour)])
#
# for d in range(len(Julianday)):
#     for t in range(len(hour)):
#         [azis[t], zens[t]] = solarpos(Julianday[d],Constants.latitude,Constants.long_rd,hour[t],radians=False)
#     plt.plot(hour,azis,label='Nov ' + str(Julianday[d]-304))
# plt.xlabel('time [h]')
# plt.legend(loc='upper right')
# plt.ylabel('azimuth angle [degrees]')
# plt.show()

