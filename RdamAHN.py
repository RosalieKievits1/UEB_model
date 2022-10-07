import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import pandas as pd
import SVF
import sys
# grid size model
xsize = 320
ysize = 320
itot = 160#64
jtot = 160#64
# xsize = 100
# ysize = 100
# itot = 25
# jtot = 25
cell_model_x = xsize/itot # in model
print('cellmodelx', cell_model_x)
cell_model_y = ysize/jtot # in model
print('cellmodely', cell_model_y)
#look at data
# data_excel = tf.imread('M_0.5_37HN1.TIF')  # dtm (topography), charlois
# plt.imshow(data_excel[6500:9700,2800:6000]) # mooie map voor M5_37HN1
data_excel = tf.imread('M_37FZ1.TIF')  # dtm (topography)
# plt.imshow(data_excel[650:650+xsize,280:280+xsize]) # mooie map voor M5_37HN1
# plt.show()

# Import data from PDOK
if cell_model_x == 5 and cell_model_y == 5:
    print('5')
    data = tf.imread('R5_37FZ1.TIF')  # dsm (all info, with building)
    data_topo = tf.imread('M5_37FZ1.TIF')  # dtm (topography)
    data[data > 10 ** 38] = 0  # remove extreme large numbers for water and set to zero.
    data_topo[data_topo > 10 ** 38] = 0

else:
    data = tf.imread('R_37FZ1.TIF')  # dsm (all info), charlois
    data_topo = tf.imread('M_37FZ1.TIF')  # dtm (topography), charlois
    #plt.imshow(data_topo[650:650 + int(xsize*10), 280:280 + int(xsize*10)])  # mooie map voor M5_37HN1
    #plt.show()
    #remove extreme large numbers for water and set zero
    data[data > 10 ** 38] = 0
    data_topo[data_topo > 10 ** 38] = 0
    '''select part of data if you want otherwise left side is taken, x_ratio * itot datapoints are needed'''
    #data = data[650:650 + int(xsize*10), 280:280 + int(xsize*10)]
    #data_topo = data_topo[650:650 + int(xsize*10), 280:280 + int(xsize*10)]

    if cell_model_x%0.5 ==0 and cell_model_y%0.5 ==0:
        x_ratio = cell_model_x#*10
        y_ratio = cell_model_y#*10
        new_data = np.zeros((itot,jtot))
        new_data_topo = np.zeros((itot,jtot))
        for i in range(itot):
            ileft = int(i*x_ratio)
            print('ileft', ileft)
            iright = int(i*x_ratio+x_ratio)-1
            print('iright', iright)
            for j in range(jtot):
                jleft = int(j * y_ratio)
                print('jleft', jleft)
                jright = int(j * y_ratio + y_ratio)-1
                print('jright', jright)
                part = data[ileft:iright, jleft:jright]
                print('part', part)
                print(np.shape(part))
                new_data[i,j] = np.mean(part)
                # print('new_data', new_data[i,j])
                new_data_topo[i,j] = np.mean(data_topo[ileft:iright, jleft:jright])
        data = new_data
        data_topo = new_data_topo
    else:
        sys.exit('take cell in model as multiple of 0.5')


[x_len, y_len] = np.shape(data)
min_height = 0.5 # buildings below 1m are not looked at.
data_diff = data - data_topo
"""Wat doet de zin hieronder???"""
#data_diff = data_diff - np.amin(data_diff)
# data_diff[data_diff > 10 ** 37] = 0  # remove extreme large numbers for water and set to zero.

"""Deletes all values below the minimum height"""
# print('datadiff', data_diff)
# for i in range(x_len):  #maybe inefficient??
#     for j in range(y_len):
#         if abs(data_diff[i,j]) < min_height:
#             data_diff[i,j] = 0


"""If all surrounding tiles are zero the middle one might be a mistake of just a lantern or something"""
for i in range(x_len):  # maybe inefficient??
    for j in range(y_len):
        if data_diff[i,j] != 0 and i < (x_len -1) and j < (y_len-1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j+1] ==0 and data_diff[i+1,j-1] ==0 and data_diff[i-1,j+1] == 0 and data_diff[i-1,j-1] ==0:
            data_diff[i, j] = 0
        elif data_diff[i,j] != 0 and i == 0 and j < (y_len-1) and data_diff[i+1,j] == 0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j+1] ==0 and data_diff[i+1,j-1] ==0:
            data_diff[i,j] = 0
        elif data_diff[i,j] != 0 and i == x_len-1 and j < (y_len-1) and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i,j-1] == 0 and data_diff[i-1,j+1] == 0 and data_diff[i-1,j-1] ==0:
            data_diff[i, j] = 0
        elif data_diff[i,j] != 0 and j == 0 and i < (x_len-1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j +1] ==0 and data_diff[i+1,j+1] ==0 and data_diff[i-1,j+1] == 0:
            data_diff[i, j] = 0
        elif data_diff[i,j] != 0 and j == (y_len-1) and i < (x_len -1) and data_diff[i+1,j] ==0 and data_diff[i-1,j] ==0 and data_diff[i,j-1] == 0 and data_diff[i+1,j-1] ==0 and data_diff[i-1,j-1] ==0:
            data_diff[i, j] = 0

"""Deletes all values below the minimum height"""
datadiffcopy = data_diff
datadiffcopy[data_diff < min_height] = 0
data_diff = datadiffcopy

# coords = SVF.coordheight(data_diff)
# SVFs = SVF.calc_SVF(coords, SVF.steps_phi, SVF.steps_theta,SVF.max_radius)
# print(SVFs)

data_diff = np.around(data_diff, 3)
print(np.shape(data_diff))
'''make ibm.inp file'''
# print('datadiff1', data_diff[670,280:600])
# print('datadiff2',data_diff[670:990,280])
# print('max datadiff', np.amax(data_diff[670:990,280:600]))
# print('min datadiff',  np.amin(data_diff[670:990,280:600]))
# plt.imshow(data_diff[670:734,280:344])
# plt.imshow(data_diff[670:990,280:600])
# plt.colorbar()
# plt.show()
# np.savetxt('ibm.inp.txt', data_diff[670:734,280:344], fmt='%1.3f') # 3 decimals
# '''Percentage of zeros in map'''
# zero_array = np.shape(np.where(data_diff[670:734,280:334]==0))[1]
# print('datadiff last version', data_diff)
# plt.imshow(data_diff)
# plt.show()

# np.savetxt('ibm.inp_new.txt', data_diff, fmt='%1.3f') # 3 decimals
# zero_array = np.shape(np.where(data_diff==0))[1]

# print(str(zero_array) , ' points are zero of the total amount of ', str(itot*jtot))
# print('This is a percentage of ', str(zero_array*100/(itot*jtot)) )

coords = SVF.coordheight(data_diff*2)
print(coords.shape[0])
SVFs = SVF.calc_SVF(coords, SVF.steps_phi, SVF.steps_theta,SVF.max_radius)
print("these are the SVF's")
print(SVFs)

#data_array = data_diff[670:734,280:344]
# print(data_array[8,14])
# print(data_array[18,25])
# print(data_array[15,45])
# print(data_array[29,14])
# print(data_array[25,35])
# print(data_array[50,55])
# print(data_array[3,2])
# print(data_array[3,41])
# print(data_array[8,4])
# print(data_array[8,28])
# print(data_array[12,37])
# print(data_array[14,4])
# print(data_array[14,49])
# print(data_array[16,33])
# print(data_array[16,48])
# print(data_array[20,15])
# print(data_array[21,52])
# print(data_array[33,25])
# print(data_array[34,29])
# print(data_array[47,13])
# print(data_array[48,31])
# print(data_array[51,11])
# print(data_array[51,48])
# print(data_array[58,50])



# np.savetxt('ibm.inp.002', data_diff[578: 578+64, 742:742+64], fmt='%1.3f') # 3 decimals
#
# '''Plot of buildings'''
# # plt.imshow(data_diff[983:1047, 191:255]) #test
# # plt.imshow(data_diff[578: 578+64, 742:742+64]) #charlois
# print('datadiff', np.shape(data_diff))
# plt.imshow(data_diff[0: 199, 0:199]) #charlois
# plt.colorbar()
# # plt.imshow(data_diff[496: 496+64, 708:708+64])
# plt.show()
#
#
# '''Write data to excel'''
#df = pd.DataFrame(data_diff[670:734,280:344])
#df.to_excel(excel_writer = "C:/Users/pamwi/OneDrive/Documenten/master/afstuderen/pycharm/heights_file.xlsx")

# df = pd.DataFrame(data_diff)
# df.to_excel(excel_writer = "C:/Users/pamwi/OneDrive/Documenten/master/afstuderen/pycharm/data_diff_charlois.xlsx")
# # print(data_build[0:10, 0:10])
# # plt.imshow(data_build)
# # plt.show()
