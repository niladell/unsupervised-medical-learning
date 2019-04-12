'''
This script serves to create a second channel to the input images from the cq500 dataset.
The first channel is the raw images and the second channel constitutes the brain window, which is fed to the model to
see if it is then possible to represent finer alterations in the brain parenchyma.
'''

import pydicom
import numpy as np
from windowing import define_window, convert_to_HU, slice_windowing

def create_channeled_image(dcm_path, window_name):
    dcm = pydicom.dcmread(dcm_path)
    dcm_HU = convert_to_HU(dcm)
    window = define_window(window_name)
    dcm_HU_wind = slice_windowing(dcm_HU, window)
    channel_dcm = np.stack((dcm.pixel_array, dcm_HU_wind))
    return channel_dcm



######################################
#      TESTING WITH AN EXAMPLE       #
######################################
if __name__ == "__main__":
    dcm = pydicom.dcmread('./test/Unknown Study/CT PLAIN THIN/CT000006.dcm')
    a = create_channeled_image('./test/Unknown Study/CT PLAIN THIN/CT000006.dcm', 'brain')