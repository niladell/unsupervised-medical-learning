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
import random
import plotly.plotly as py
import plotly.graph_objs as go


path2 = '/Users/ines/Desktop/subjects copy'
os.chdir(path2)

##################
# TRYING OUT PCA #
##################


# Loop to create an list with the pixel data per slice
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

x_array = np.asarray(x) # convert to np.array for ease of manipulation
id_array = np.asarray(id)

## With Tensorflow --> did not work
# # Perform SVD
# singular_values, u, _ = tf.svd(list_tensors)
# # Create sigma matrix
# sigma = tf.diag(singular_values)

## With sklearn

# If available, upload data (issue with gdcm and MacOs)
x_array = np.load('pca_experiments_dcms_pix.npy')
id_array = np.load('pca_experiments_dcms_id.npy')

# Preprocessing:


x_reshaped = x_array.reshape(x_array.shape[0], 512*512) # Reshape so you can run sk-learn pca
# This particular reshaping is inspired by:
# https://stackoverflow.com/questions/48003185/sklearn-dimensionality-issues-found-array-with-dim-3-estimator-expected-2

# # Normalization (not used, for now)
# sc = StandardScaler()
# x_normalized = sc.fit_transform(x_reshaped)

# Performing PCA
pca = PCA(n_components=3)
# pca.fit(pixel_stuff) # has to have 2 or less dimensions

x_train = pca.fit_transform(x_reshaped)

principalDf = pd.DataFrame(data = x_train, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# Visualization (commented out code leads to 3D visualization)
# Tutorial on 3D visualization in matplotlib: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

# number_of_colors = len(id_array)
#
# colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#              for i in range(number_of_colors)]
# # print(colors)

# Function (courtesy of https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python)
def define_colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r,g,b))
  return ret


colors = define_colors(len(set(id_array)))
# colors = np.array(colors)
# colors = np.reshape(colors, 14*3)


# Plotting
fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
set_ids = set(id_array)
for iden, color in zip(set_ids, colors):
    idx = id_array == iden
    ax.scatter(principalDf.loc[idx, 'principal component 1'],
               principalDf.loc[idx, 'principal component 2'],
               principalDf.loc[idx, 'principal component 3'],
               cmap=color,
               )
ax.legend(set_ids)
ax.grid()
plt.show()

# Interactive plotting:
# https://plot.ly/python/line-and-scatter/

x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
for iden, color in zip(set_ids, colors):
    idx = id_array == iden
    trace1 = go.Scatter3d(
        x=principalDf.loc[idx, 'principal component 1'],
        y=principalDf.loc[idx, 'principal component 2'],
        z=principalDf.loc[idx, 'principal component 3'],
        mode='markers',
        marker=dict(
            size=12,
            line=dict(
                color=color,
                width=0.5
            ),
            opacity=0.8
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')

# https://bokeh.pydata.org/en/latest/docs/gallery/iris_splom.html

# Trying to reconstruct the image
approximation = pca.inverse_transform(x_train)


