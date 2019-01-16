'''
Interesting resources:
- https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
- Getting the inverse transform: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb
-

Thoughts on PCA results:
- Subjects 8 and 100 are separated from the rest of the lot, along the PC1 axis!
- So what does PC1 encode??
    - bear in mind that subjects
    - Number 8 has a big yaw (the head is quite tilted)

'''

import matplotlib.pyplot as plt
# from pydicom.data import get_testdata_files
import os
import pdb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# from PIL import Image
# import PIL
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
plotly.tools.set_credentials_file(username='InesPereira', api_key='wvXCCdZJ04XVr5mDhjn3')
import pickle
import time



def generate_image_and_id_arrays(path):
    # Loop to create an list with the pixel data per slice
    x = []
    id = []
    for file in tqdm(os.listdir(path)):
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

    return x_array, id_array

def PCA_function(x_array, pca):
    print('Starting PCA on the given data...')
    # Preprocessing:
    x_reshaped = x_array.reshape(x_array.shape[0], 512*512) # Reshape so you can run sk-learn pca
    # This particular reshaping is inspired by:
    # https://stackoverflow.com/questions/48003185/sklearn-dimensionality-issues-found-array-with-dim-3-estimator-expected-2

    # Normalization (recommended everywhere)
    sc = StandardScaler()
    x_reshaped = sc.fit_transform(x_reshaped)
    print('Normalization of the data done!')

    # Check out how many components are necessary to explain the above defined variance:
    # pca.n_components_

    x_train = pca.fit_transform(x_reshaped)
    print('Amazing! Your PCA has been completed!')

    return x_train

# Visualization (commented out code leads to 3D visualization)
# Tutorial on 3D visualization in matplotlib: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

# number_of_colors = len(id_array)
#
# colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#              for i in range(number_of_colors)]
# # print(colors)

# Function (courtesy of https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python)
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


def interactive_plotting(principalDf, id_array):
    # Interactive plotting with plotly:
    colors = define_colors(len(set(id_array)))
    set_ids = set(id_array)
    data_full =[]
    for iden, color in zip(set_ids, colors):
        idx = id_array == iden
        trace1 = go.Scatter3d(
            x=principalDf.loc[idx, 'principal component 1'],
            y=principalDf.loc[idx, 'principal component 2'],
            z=principalDf.loc[idx, 'principal component 3'],
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
            xaxis=XAxis(title='Principal Component 1'),
            yaxis=YAxis(title='Principal Component 2'),
            zaxis=ZAxis(title='Principal Component 3')
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='PCA with 3 components')


# Another library for interactive plots: https://bokeh.pydata.org/en/latest/docs/gallery/iris_splom.html


if __name__ == "__main__":

    path1 = os.path.join(os.path.dirname(__file__), 'dcms_pix.npy')
    path2 = os.path.join(os.path.dirname(__file__), 'dcms_id.npy')
    x_array = np.load(path1)
    print('Data array loaded.')
    print(x_array.shape)
    id_array = np.load(path2)
    print('ID array loaded.')

    # If available, upload data (issue with gdcm and MacOs).
    # Otherwise run the generate_image_and_id_arrays(path) function
    # x_array = np.load('src/util/dcms_pix.npy')
    # print('Data array loaded.')
    # id_array = np.load('src/util/dcms_id.npy')
    # print('ID array loaded.')


    # Performing PCA
    n_components = 2
    pca = PCA(n_components=n_components)
    pca = PCA(.95) # It means that scikit-learn choose the minimum number of principal components
    # such that 95% of the variance is retained.

    x_train = PCA_function(x_array, pca)

    # These things are computationally expensive, so save your models!
    filename = 'pca_'+str(pca.n_components)+'PC_'+str(time.strftime("%d_%m_%Y"))+'_'+str(time.strftime("%H:%M:%S"))+'.pickle'
    pickle.dump(pca, open(filename, 'wb'))

    # Create a dataframe for all the data:
    columns = []
    for i in range(1, n_components+1):
        string = 'principal component '+ str(i)
        columns.append(string)

    principalDf = pd.DataFrame(data=x_train, columns=columns)

    static_plotting(principalDf, id_array)
    interactive_plotting(principalDf, id_array)

    # Trying to reconstruct the image
    # Thanks to: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb
    approximation = pca.inverse_transform(x_train)

    # Plotting
    plt.figure(figsize=(8, 4));
    # Original Image
    plt.subplot(1, 2, 1);
    plt.imshow(x_array[0, :, :].reshape(512, 512),
               cmap=plt.cm.gray, interpolation='nearest');
    # plt.xlabel('?? components', fontsize=14)  # How many components do we really have?
    plt.title('Original Image', fontsize=20);
    plt.show()


    # ??? principal components
    plt.subplot(1, 2, 2);
    plt.imshow(approximation[0].reshape(512, 512),
               cmap=plt.cm.gray, interpolation='nearest');
    plt.xlabel('?? components', fontsize=14)
    plt.title('95% of Explained Variance', fontsize=20);
    plt.show()

    # Trying to get the meaning out the principal components:
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    # Todo:
    # - try out different types of PCA (KernelPCA, SparsePCA, TruncatedSVD, IncrementalPCA)
    # - run the computationally more expensive things on Euler


    #######################################
    # WORKING ON PCA COMPUTED ON LEONHARD #
    #######################################
    filename = 'pca_0.95PC_13_01_2019_18:19:42.pickle'
    pca_model = pickle.load(open('src/util/'+filename, 'rb'))
    n_components = pca_model.n_components_

    # Perform PCA transformation on data
    x_reshaped = x_array.reshape(x_array.shape[0], 512*512)
    x_transformed = pca_model.transform(x_reshaped)

    # Create a dataframe for all the data:
    columns = []
    for i in range(1, n_components+1):
        string = 'principal component '+ str(i)
        columns.append(string)

    principalDf = pd.DataFrame(data=x_transformed, columns=columns)

    # Plotting the transformed data:
    static_plotting(principalDf, id_array)
    interactive_plotting(principalDf, id_array)

    # Creating approximations with all components:
    approximation = pca_model.inverse_transform(x_transformed)
    slice_approx = approximation[10,:] # pick a slice to plot
    approx_reshaped = slice_approx.reshape(512, 512)  # reshape so you get back your image :)
    greyscale_plot(approx_reshaped)


    # Trying to isolate single PC:
    # https://stackoverflow.com/questions/32750915/pca-inverse-transform-manually
    PC1 = pca_model.components_[0,:]
    U_reduced = np.zeros(pca_model.components_.shape)  # Creating a new U matrix which will have only one PC
    U_reduced[0,:] = PC1  # here we let U_reduced have only PC1
    PC1_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_  # recreating approximation with only PC1
    # Save your arrays and spare your computer :)
    np.save('src/util/PC1_approximation.npy', PC1_approx)
    greyscale_plot(PC1_approx[10,:].reshape(512, 512))  # resembles an x-ray

    PC2 = pca_model.components_[1,:]
    U_reduced = np.zeros(pca_model.components_.shape)
    U_reduced[1,:]=PC2
    PC2_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # Save your arrays and spare your computer :)
    np.save('src/util/PC2_approximation.npy', PC2_approx)
    greyscale_plot(PC2_approx[12,:].reshape(512, 512))  # maybe encoding pixel values?

    PC3 = pca_model.components_[2,:]
    U_reduced = np.zeros(pca_model.components_.shape)
    U_reduced[2,:]=PC3
    PC3_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # Save your arrays and spare your computer :)
    np.save('src/util/PC3_approximation.npy', PC3_approx)
    greyscale_plot(PC3_approx[12,:].reshape(512, 512))

    # Now let's combine them
    U_reduced = np.zeros(pca_model.components_.shape)
    U_reduced[0,:]=PC1
    U_reduced[1,:]=PC2
    PC12_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    greyscale_plot(PC12_approx[12,:].reshape(512, 512))

    U_reduced = np.zeros(pca_model.components_.shape)
    U_reduced[0,:] = PC1
    U_reduced[1,:] = PC2
    U_reduced[2,:] = PC3
    PC123_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    greyscale_plot(PC123_approx[12,:].reshape(512, 512))

    # Removing components:
    U_reduced = pca_model.components_
    U_reduced[2,:] = 0
    PC_less1_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    greyscale_plot(PC_less1_approx[10,:].reshape(512, 512))









