#!/usr/bin/env python3

"""
Reads data from pickle data and performs tsne
"""

import os
import pdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pydicom
import matplotlib.pyplot as plt
import re
# import gdcm
from tqdm import tqdm
import random
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
plotly.tools.set_credentials_file(username='pjaweh', api_key='dD94S722iwnqJgUV51qb')
import pickle
import time
import argparse
from sklearn.manifold import TSNE


def get_slices_from_pickle(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
        return data

def get_labels(dictionary):
    id_array = []
    for subj in dictionary:
        pattern = re.compile(r'CQ500CT\d+\s')
        m = re.search(pattern, subj)
        id = (m.group(0))
        id = len(subj)*[id]
        id_array.extend(id)
    return id_array

def generate_image_array_from_pickle(dictionary):
    x_array = []
    id_array = []
    for subj in dictionary:
        cnt = 0
        for img_array in dictionary[subj]:
            if img_array.shape == (512,512):
                x_array.append(img_array)
                cnt +=1
        pattern = re.compile(r'CQ500CT\d+\s')
        m = re.search(pattern, subj)
        id = (m.group(0))
        id = cnt*[id]
        id_array.extend(id)
    x_array = np.asarray(x_array)
    id_array = np.asarray(id_array)
    return x_array, id_array

def generate_image_and_id_arrays(list_of_dcms):
    # Loop to create a list with the pixel data per slice
    x = []
    ids = []
    # for file in tqdm((list_of_paths)):
    for file in tqdm(list_of_dcms):
        pattern = re.compile(r'.dcm')
        m = re.search(pattern, file)
        if m is not None:
            # print(file)
            dcm = pydicom.dcmread(file)
            identity = dcm.get('PatientID')
            if dcm.pixel_array.shape == (512, 512):
                x.append(dcm.pixel_array)
                ids.append(identity)

    x_array = np.asarray(x) # convert to np.array for ease of manipulation
    id_array = np.asarray(ids)

    return x_array, id_array


def tsne_function(x_array):
    x_reshaped = x_array.reshape(x_array.shape[0], 512*512)
    n_sne = 7000
    
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    x_train = tsne.fit_transform(x_reshaped)
    
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return x_train, tsne

def greyscale_plot(object):
    plt.figure();
    plt.imshow(object, cmap=plt.cm.gray, interpolation='nearest');
    plt.show()

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


def static_plotting(principalDf, id_array):
    # Defining colors automatically:
    colors = define_colors(len(set(id_array)))

    # Plotting
    fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(1,1,1)
    ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('tsne Component 1', fontsize = 15)
    ax.set_ylabel('tsne Component 2', fontsize = 15)
    ax.set_zlabel('tsne Component 3', fontsize = 15)
    ax.set_title('3 component tsne', fontsize = 20)
    set_ids = set(id_array)
    for iden, color in zip(set_ids, colors):
        idx = id_array == iden
        ax.scatter(principalDf.loc[idx, 'tsne component 1'],
                   principalDf.loc[idx, 'tsne component 2'],
                   principalDf.loc[idx, 'tsne component 3'],
                   cmap=color,
                   )
    ax.legend(set_ids)
    ax.grid()
    plt.show()


def interactive_plotting(principalDf, id_array):
    # Interactive plotting with plotly:
    colors = define_colors(len(set(id_array)))
    set_ids = set(id_array)
    data_full =[]
    for iden, color in zip(set_ids, colors):
        idx = id_array == iden
        trace1 = go.Scatter3d(
            x=principalDf.loc[idx, 'tsne component 1'],
            y=principalDf.loc[idx, 'tsne component 2'],
            z=principalDf.loc[idx, 'tsne component 3'],
            name=iden,
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
        data_full.append(trace1)

    data = data_full
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0),
        scene=Scene(
            xaxis=XAxis(title='tsne Component 1'),
            yaxis=YAxis(title='tsne Component 2'),
            zaxis=ZAxis(title='tsne Component 3')
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='TSNE with 3 components')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Renormalize slice locations  \
        and reorder filenames according to slice heigth')

    parser.add_argument('-p', '--path',
                        help='Path where to start searching', default='.')

    args = parser.parse_args()
    btm_dict = get_slices_from_pickle('btm_slices.p')
    x_array, id_array = generate_image_array_from_pickle(btm_dict)
    print(id_array.shape)
    print(x_array.shape)

    x_train, tsne = tsne_function(x_array)
    print('Training complete.')

#    # store model
#    filename = 'tsne_'+str(tsne.n_components)+'PC_'+str(time.strftime("%d_%m_%Y"))+'_'+str(time.strftime("%H:%M:%S"))+'.pickle'
#    pickle.dump(pca, open(filename, 'wb'))
#    print('Model saved!')

#    print('Creating a dataframe to plot the data in 3D...')
    principalDf = pd.DataFrame(data = x_train, columns = ['tsne component 1', 'tsne component 2', 'tsne component 3'])

#    #static_plotting(principalDf, id_array)
    interactive_plotting(principalDf, id_array)