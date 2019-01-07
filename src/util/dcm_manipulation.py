import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import os
import tensorflow as tf
import pdb
import numpy as np
from sklearn.decomposition import PCA

tf.enable_eager_execution()


path='/Users/ines/Desktop/subjects/subject108/Unknown Study/CT 0.625mm'
os.chdir(path)

filename = 'CT000000.dcm'

ds = pydicom.dcmread(filename) # direct conversion of ds to tensor does not work!

######################################
# PLAYING AROUND WITH PATRICK'S CODE #
######################################

from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import re

path2 = '/Users/ines/Desktop/pca_experiments'
os.chdir(path2)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


tfrecords_outfile = 'test.tfrecords'


filename = 'CT000000.dcm'
ds = pydicom.dcmread(filename)
print(ds)

pixel_data = []
for file in os.listdir(path2):
    pattern = re.compile(r'.dcm$')
    m = re.search(pattern, file)
    if m is not None:
        # writer = tf.python_io.TFRecordWriter(tfrecords_outfile)
        ds = pydicom.dcmread(file)
        img_raw = ds.PixelData
        height = ds.pixel_array.shape[0]
        width = ds.pixel_array.shape[1]
        location = ds.get('SliceLocation', "(missing)")
        identity = ds.get('PatientID')
        identity_nr = re.findall(r'\d+$', identity)

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'identity_nr': _int64_feature(int(identity_nr[0])),
            'slice_location': _float_feature(location)}))

        # save pixel information to list
        pixel_data.append(img_raw)

        # # print(example)
        # writer.write(example.SerializeToString())
        #
        # writer.close()

##################
# TRYING OUT PCA #
##################



x = []
for file in os.listdir(path2):
    pattern = re.compile(r'.dcm$')
    m = re.search(pattern, file)
    if m is not None:
        dcm = pydicom.dcmread(file)
        x.append(dcm.pixel_array)

# tft.pca(x, output_dim=4, dtype=tf.float64)

tensor = tf.convert_to_tensor(example) # this also does not work

tensor = tf.convert_to_tensor(img_raw) # works!


list_tensors = []
for data in pixel_data:
    tensor = tf.convert_to_tensor(data) # works?
    list_tensors.append(tensor)

## With Tensorflow --> did not work
# Perform SVD
singular_values, u, _ = tf.svd(list_tensors)
# Create sigma matrix
sigma = tf.diag(singular_values)

## With sklearn
pixel_stuff = pixel_data[0:2]
pca = PCA(n_components=5)
pca.fit(pixel_stuff)

x_array = np.asarray(x)
x_reshaped = x_array.reshape(36, 512*512) # This particular reshaping is inspired by:
# https://stackoverflow.com/questions/48003185/sklearn-dimensionality-issues-found-array-with-dim-3-estimator-expected-2
pca.fit(x_reshaped)

