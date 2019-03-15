""" Set of functions to adapt the data, as croping, resizing, formatting, etc.

Some functions related to downloading and adapting the celebA dataset have been modified from 'nmhkahn/DCGAN-tensorflow-slim' (Github repo)
"""

import scipy
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import re
from scripts.retrieve_all_dcms import list_forbidden_folders
from src.util.windowing import slice_windowing, win_dict



def center_crop( im,
                output_height,
                output_width ):
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("The image is too small ({}, {}) for that size ({}, {})"\
                    .format(h, w, output_height, output_width))

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    return im[offset_h:offset_h+output_height,
                offset_w:offset_w+output_width, :]

re_forbidden_folders = re.compile(r'\b(?:%s)\b' % '|'.join(list_forbidden_folders))


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


def convert_to_HU(pixel_array, slope, intercept):
    HU = pixel_array*slope + intercept
    return HU

def define_window(window_name):
    wl = win_dict[window_name]['wl']
    ww = win_dict[window_name]['ww']
    max_window = wl + ww
    min_window = wl - ww
    return [min_window, max_window]


######################################
#      TESTING WITH AN EXAMPLE       #
######################################

dcm = pydicom.dcmread('/Users/ines/Desktop/test/Unknown Study/CT PLAIN THIN/CT000006.dcm')

new_wind_dcm = np.load('/Users/ines/Downloads/windowed_dcm.npy')
dcm_HU = np.load('/Users/ines/Downloads/dcm_HU.npy')

plt.figure();
plt.imshow(new_wind_dcm, cmap=plt.cm.gray, interpolation='nearest');
plt.show()

plt.figure();
plt.imshow(dcm_HU, cmap=plt.cm.gray, interpolation='nearest');
plt.show()



