import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import csv
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
    hour_sunrise = (omega_sunrise/0.261799+12)-EOT-(longitude-long_tz)/15
    hour_sunset =  (omega_sunset/0.261799+12)-EOT-(longitude-long_tz)/15

    """the sun change 15 degrees or 0.26 radians per hour"""
    hour_rad = (T-12)*0.261799
    """the declination angle (acitually the zenith is defined as angle from top so not the same)"""
    el_angle = np.arcsin(np.sin(delta)*np.sin(latitude)+np.cos(delta)*np.cos(latitude)*np.cos(hour_rad))

    azimuth = np.arcsin(np.cos(delta)*np.sin(hour_rad)/np.cos(el_angle))
    if (np.cos(hour_rad)<(np.tan(delta)/np.tan(latitude))):
        if T<12:
            azimuth = abs(azimuth)+np.pi
        elif T>12:
            azimuth = np.pi-azimuth

    """Correct to measure from north"""
    azimuth = (azimuth+np.pi)%(2*np.pi)

    """Transform to degrees if we set the boolean radians to false"""
    """The azimuth angle is 0 when the sun is south and positive westwards"""
    if (radians==False):
        azimuth = azimuth*180/np.pi
        azimuth = azimuth % 360
        el_angle = el_angle*180/np.pi

    return azimuth,el_angle


# df = pd.read_csv('SolarPosMay1.csv')
# print(df)

# with open("SolarPosMay1.csv", 'r') as file:
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
#
# #Julianday = np.linspace(305,311,7,dtype=int) # nov 1
Julianday = np.linspace(121,128,7,dtype=int) # may 1

# plt.figure()
#
# [azis_o, zens_0,T_sunrise,T_sunset] = solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,1,radians=False)
#
# hour = np.linspace(T_sunrise,T_sunset,100)
# hour_r = np.linspace(0,24,25)
# print(T_sunrise)
# print(T_sunset)
# azis = np.zeros([len(hour)])
# zens = np.zeros([len(hour)])
#for d in range(len(Julianday)):
#print(solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,T_sunrise,radians=False))
# for t in range(len(hour)):
#     [azis[t], zens[t],Tsr,Tss] = solarpos(Constants.julianday,Constants.latitude,Constants.long_rd,hour[t],radians=False)
# plt.plot(hour,zens,label='Computed Elevation Angle')
# plt.plot(hour_r[5:21],a[5:21],label='Measured Elevation Angle')
# plt.legend(loc='upper right')
# #,label=str(Constants.julianday-304) + ' Nov')
# plt.xlabel('time [h]')
# #plt.legend(loc='upper left')
# plt.ylabel('Elevation angle [degrees]')
# plt.show()

# hour = np.linspace(T_sunrise,T_sunset,50)
# #Julianday = np.linspace(305,312,8,dtype=int)
# Julianday = np.linspace(121,128,8,dtype=int)
#
# plt.figure()
# azis_0 = np.zeros([len(hour)])
# zens_0 = np.zeros([len(hour)])
# azis = np.zeros([len(hour)])
# zens = np.zeros([len(hour)])
#
# for d in range(len(Julianday)):
#     for t in range(len(hour)):
#         [azis_0[t], zens_0[t],T_sr,T_ss] = solarpos(121,Constants.latitude,Constants.long_rd,hour[t],radians=False)
#         [azis[t], zens[t],T_sr,T_ss] = solarpos(Julianday[d],Constants.latitude,Constants.long_rd,hour[t],radians=False)
#     plt.plot(hour,zens-zens_0,label='May ' + str(Julianday[d]-120))
# plt.xlabel('time [h]')
# plt.legend(loc='upper right')
# plt.ylabel('difference in elevation angle [degrees]')
# plt.show()

