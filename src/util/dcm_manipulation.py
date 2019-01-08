import matplotlib.pyplot as plt
# from pydicom.data import get_testdata_files
import os
import tensorflow as tf
import pdb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# from PIL import Image
# import PIL
import numpy as np
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import re
# import gdcm
from tqdm import tqdm


# tf.enable_eager_execution()


# path='/Users/ines/Desktop/subjects/subject108/Unknown Study/CT 0.625mm'
# os.chdir(path)

# filename = 'CT000000.dcm'
#
# ds = pydicom.dcmread(filename) # direct conversion of ds to tensor does not work!

######################################
# PLAYING AROUND WITH PATRICK'S CODE #
######################################


path2 = '/Users/ines/Desktop/subjects copy'
os.chdir(path2)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


tfrecords_outfile = 'test.tfrecords'


# filename = 'CT000000.dcm'
# ds = pydicom.dcmread(filename)
# print(ds)
#
# pixel_data = []
# for file in os.listdir(path2):
#     pattern = re.compile(r'.dcm$')
#     m = re.search(pattern, file)
#     if m is not None:
#         # print(file)
#         # writer = tf.python_io.TFRecordWriter(tfrecords_outfile)
#         ds = pydicom.dcmread(file)
#         img_raw = ds.PixelData
        # height = ds.pixel_array.shape[0]
        # width = ds.pixel_array.shape[1]
        # location = ds.get('SliceLocation', "(missing)")
        # identity = ds.get('PatientID')
        # identity_nr = re.findall(r'\d+$', identity)

        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'height': _int64_feature(height),
        #     'width': _int64_feature(width),
        #     'image_raw': _bytes_feature(img_raw),
        #     'identity_nr': _int64_feature(int(identity_nr[0])),
        #     'slice_location': _float_feature(location)}))

        # save pixel information to list
        # pixel_data.append(img_raw)

        # # print(example)
        # writer.write(example.SerializeToString())
        #
        # writer.close()

##################
# TRYING OUT PCA #
##################



x = []
id = []
for file in tqdm(os.listdir(path2)):
    pattern = re.compile(r'.dcm$')
    m = re.search(pattern, file)
    if m is not None:
        # print(file)
        dcm = pydicom.dcmread(file)
        identity = dcm.get('PatientID')
        x.append(dcm.pixel_array)
        id.append(identity)

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
# Preprocessing:
x_array = np.asarray(x) # convert to np.array for ease of manipulation
id_array = np.asarray(id)

x_reshaped = x_array.reshape(x_array.shape[0], 512*512) # Reshape so you can run sk-learn pca
# This particular reshaping is inspired by:
# https://stackoverflow.com/questions/48003185/sklearn-dimensionality-issues-found-array-with-dim-3-estimator-expected-2

#Small experiment
id_array = np.array([2,3,4,5])
id_reshaped = id_array.reshape((id_array.shape[0],1,1))
id_repeat = np.repeat(id_reshaped, 5, axis=1)
id_repeat = np.repeat(id_repeat, 5, axis=2)
id_reshaped = id_repeat.reshape(id_repeat.shape[0], 5*5)

id_reshaped = id_array.reshape((id_array.shape[0],1,1))
id_repeat = np.repeat(id_reshaped, 512, axis=1)
id_repeat = np.repeat(id_repeat, 512, axis=2)
id_reshaped = id_repeat.reshape(x_array.shape[0], 512*512)


# Normalization
sc = StandardScaler()
x_normalized = sc.fit_transform(x_reshaped)

# Performing PCA
pca = PCA(n_components=3)
# pca.fit(pixel_stuff) # has to have 2 or less dimensions

x_train = pca.fit_transform(x_reshaped)

principalDf = pd.DataFrame(data = x_train, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# Visualization (commented out code leads to 3D visualization)
# Tutorial on 3D visualization in matplotlib: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax = plt.axes(projection='3d')
ax = fig.add_subplot(111) #, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
set_ids = set(id)
colors = ['blue', 'black']
for iden, color in zip(set_ids, colors):
    idx = id_array == iden
    ax.scatter(principalDf.loc[idx, 'principal component 1'],
               principalDf.loc[idx, 'principal component 2'],
               color=color
               )#, principalDf.loc[:, 'principal component 3'])
ax.grid()
plt.show()

# Trying to reconstruct the image
approximation = pca.inverse_transform(x_train)


