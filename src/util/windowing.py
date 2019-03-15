'''
Script to extract windows from the generated images.
The generated images have pixel values between -1 and 1.

The assumptions that we make are:
- we only have cortical bone, which has +1800 to +1900 HU (https://en.wikipedia.org/wiki/Hounsfield_scale)
    - we set this value to be 1 in our scale
- Air corresponds to -1000 and will be black, hence, we correspond it to -1.

'''
from PIL import Image
import numpy as np
# import skimage.io as io
import tensorflow as tf
import pydicom
import os
import matplotlib.pyplot as plt
from scripts.retrieve_all_dcms import list_forbidden_folders
import re


# dcm_array = np.load('dcms_pix.npy')

# slice = dcm_array[0, :, :]

# ###################################################################
# # Getting all the slopes and intercepts for a bunch of dcm slices #
# ###################################################################
#
# # list_forbidden_folders.append('CT BRAIN PLAIN')
# # list_forbidden_folders.append('CT Thin PLAIN')
# # list_forbidden_folders.append('CT PLAIN')
# re_forbidden_folders = re.compile(r'\b(?:%s)\b' % '|'.join(list_forbidden_folders))
#
#
# slopes = []
# intercepts = []
#
# def get_dcms(path):
#     for dirpath, dirname, filenames in os.walk(path):
#         for file in filenames:
#             pattern = re.compile(r'.dcm$')
#             m = re.search(pattern, file)
#             if m is not None and re_forbidden_folders.search(dirpath) is None:
#                 print(dirpath+'/'+file)
#                 dcm = pydicom.dcmread(dirpath+'/'+file)
#                 slope = int(dcm.data_element('RescaleSlope').value)
#                 intercept = int(dcm.data_element('RescaleIntercept').value)
#                 slopes.append(slope)
#                 intercepts.append(intercept)
#
#
#
# path = '/Users/ines/Desktop/subjects copy/'
# get_dcms(path)
#
# mylist1 = list(set(slopes))  # o valor parece ser sempre 1
# mylist2 = list(set(intercepts))  # o valor parece ser sempre -1024
#
# #########################################################

# Not yet used

win_dict = {
            'air':
                {'wl':-1000, 'ww':0},
            'subdural':
                {'wl': 75, 'ww': 280},
            'angio':
                {'wl': 300, 'ww': 600},
            'bone':
                {'wl': 600, 'ww': 2800},
            'brain':
                {'wl': 40, 'ww': 80},
            'stroke':
                {'wl': 40, 'ww': 40},
            'soft tissues':
                {'wl': 40, 'ww': 375}}

###########################################################

def pixel_windowing(rescaled_pixel_value, window):
    max_density = window[1]
    min_density = window[0]
    if (rescaled_pixel_value == 1) or (rescaled_pixel_value>=max_density):
        new_value = max_density
    elif (rescaled_pixel_value == -1) or (rescaled_pixel_value <= min_density):
        new_value = min_density
    else:
        alpha = (rescaled_pixel_value - min_density)/(max_density-min_density)
        # new_value = rescaled_pixel_value*((max_density-min_density)/2)
        # new_value = rescaled_pixel_value - (rescaled_pixel_value - min_density)*min_density/max_density
        new_value = (alpha*max_density) + (1-alpha)*min_density
    return new_value


def slice_windowing(slice, window):
    new_data = np.empty((512, 512))
    for i in range(512):
        for j in range(512):
            new_data[i,j]=pixel_windowing(slice[i,j], window)
    return new_data

#
# slice[slice == -2000] = 0
# slice_rescaled = (slice/slice.max())*2 - 1
#
# # View slice_rescaled value distributions:
# x = slice_rescaled.flatten()
#
# # Plot of the rescaled image, before applying windowing:
# plt.hist(x, bins=40)
# plt.show()
#
# # Applying windowing:
# windowed_rescaled_slice = slice_windowing(slice= slice_rescaled, window=[-0.3, -0.2])

'''
List of good brain windows:
    - dcm_array[0, :, :] : window=[-0.21, -0.18]
    - 
'''
#
# # Rescaled image:
# plt.figure();
# plt.imshow(slice_rescaled, cmap=plt.cm.gray, interpolation='nearest');
# plt.show()
#
# # Rescaled and windowed image:
# plt.figure();
# plt.imshow(windowed_rescaled_slice, cmap=plt.cm.gray, interpolation='nearest');
# plt.show()

# import scipy.io
# scipy.io.savemat('something.mat', mdict={'arr': windowed_rescaled_slice, 'orig_arr': slice_rescaled})


plt.figure();
plt.imshow(new_wind_dcm, cmap=plt.cm.gray, interpolation='nearest');
plt.show()


