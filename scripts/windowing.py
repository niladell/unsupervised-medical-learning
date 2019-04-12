'''
Script to extract windows from the generated images.
The generated images have pixel values between -1 and 1.

The assumptions that we make are:
- we only have cortical bone, which has +1800 to +1900 HU (https://en.wikipedia.org/wiki/Hounsfield_scale)
    - we set this value to be 1 in our scale
- Air corresponds to -1000 and will be black, hence, we correspond it to -1.

'''

import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import re
from retrieve_all_dcms import list_forbidden_folders

re_forbidden_folders = re.compile(r'\b(?:%s)\b' % '|'.join(list_forbidden_folders))


# Dictionary with the most relevant windows:
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

slopes = []
intercepts = []
def get_dcms(path):
    for dirpath, dirname, filenames in os.walk(path):
        print(filenames)
        for file in filenames:
            pattern = re.compile(r'.dcm$')
            m = re.search(pattern, file)
            if m is not None and re_forbidden_folders.search(dirpath) is None:
                print(dirpath+'/'+file)
                dcm = pydicom.dcmread(dirpath+'/'+file)
                slope = int(dcm.data_element('RescaleSlope').value)
                intercept = int(dcm.data_element('RescaleIntercept').value)
                slopes.append(slope)
                intercepts.append(intercept)


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


def convert_to_HU(dcm):
    HU = dcm.pixel_array*dcm.RescaleSlope + dcm.RescaleIntercept
    return HU

def define_window(window_name):
    # https://radiopaedia.org/articles/windowing-ct?lang=gb.
    wl = win_dict[window_name]['wl']
    ww = win_dict[window_name]['ww']
    max_window = wl + ww/2
    min_window = wl - ww/2
    return [min_window, max_window]


######################################
#      TESTING WITH AN EXAMPLE       #
######################################
if __name__ == "__main__":
    dcm = pydicom.dcmread('/Users/ines/Desktop/test/Unknown Study/CT PLAIN THIN/CT000006.dcm')

    new_wind_dcm = np.load('/Users/ines/Downloads/windowed_dcm.npy')
    new_wind_dcm_half = np.load('/Users/ines/Downloads/win_HU_half.npy')

    dcm_HU = np.load('/Users/ines/Downloads/dcm_HU.npy')

    plt.figure()
    plt.imshow(new_wind_dcm_half, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()

    plt.figure()
    plt.imshow(dcm_HU, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()