'''
Interesting resources:
- https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
- Getting the inverse transform: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb
-

Thoughts on PCA results:
- Subjects 8 and 100 are separated from the rest of the lot, along the PC1 axis!
- So what does PC1 encode??
    - bear in mind that subjects 0 and 100 don't have the component at all!
    - Number 8 has a big yaw (the head is quite tilted)
- We reconstructing the images using individual PC, you only start to see differences between slices after combining at
least 3 components!

- Variance explained by PC:
    - PC1: 0.22435345
    - PC2: 0.12327218
    - PC3: 0.0656076
    - PC4: 0.04517845

CONCLUSION ON THIS 14 SUBJECT DATASET: PCA DOES NOT SEEM TO ENCODE HIGH LEVEL CHARACTERISTICS!

    # Todo:
    # - try out different types of PCA (KernelPCA, SparsePCA, TruncatedSVD, IncrementalPCA)
    # - run the computationally more expensive things on Euler

'''

# from pydicom.data import get_testdata_files
import os
import pdb
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
import pickle
import time

# list_forbidden_folders = ['CT 4cc sec 150cc D3D on',
#                           'CT 4cc sec 150cc D3D on-2',
#                           'CT 4cc sec 150cc D3D on-3',
#                           'CT POST CONTRAST',
#                           'CT POST CONTRAST-2',
#                           'CT BONE',
#                           'CT I To S',
#                           'CT PRE CONTRAST BONE',
#                           'CT Thin Bone',
#                           'CT Thin Stnd',
#                           'CT 0.625mm',
#                           'CT 0.625mm-2',
#                           'CT 5mm POST CONTRAST',
#                           'CT ORAL IV',
#                           'CT 55mm Contrast',
#                           'CT BONE THIN',
#                           'CT 3.753.75mm Plain',
#                           'CT Thin Details',
#                           'CT Thin Stand']
#
# re_forbidden_folders = re.compile(r'\b(?:%s)\b' % '|'.join(list_forbidden_folders))
#
# def get_dcms(path):
#     list_of_dcm = []
#     for dirpath, dirname, filenames in os.walk(path):
#         for file in filenames:
#             pattern = re.compile(r'.dcm$')
#             m = re.search(pattern, file)
#             if m is not None and re_forbidden_folders.search(dirpath) is None:
#                 dcm_path = dirpath + '/' + file
#                 list_of_dcm.append(dcm_path)
#     return list_of_dcm


def generate_image_and_id_arrays(list_of_paths):
    # Loop to create a list with the pixel data per slice
    x = []
    id = []
    # for file in tqdm((list_of_paths)):
    list_of_dcms = list_of_dcms[0:11]
    for file in tqdm(list_of_dcms):
        pattern = re.compile(r'.dcm$')
        m = re.search(pattern, file)
        if m is not None:
            # print(file)
            dcm = pydicom.dcmread(file)
            identity = dcm.get('PatientID')
            if dcm.pixel_array.shape == (512, 512):
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

    # YOU HAVE TO BE IN THE DATASET FOLDER TO RUN THIS SCRIPT

    # os.chdir('/Users/ines/Dropbox/CT_head_trauma')
    # list_of_dcms = get_dcms('/Users/ines/Dropbox/CT_head_trauma')

    with open('list_of_dcms.txt', 'r') as f:
        list_of_dcms = f.read().splitlines()

    x_array, id_array = generate_image_and_id_arrays(list_of_dcms)
    print('x_array and id_arrays have been generated!')
    np.save('x_array.npy', x_array)
    print('x_array saved in a npy file.')
    np.save('id_array.npy', id_array)
    print('id_array saved in a npy file.')

    # path1 = os.path.join(os.path.dirname(__file__), 'dcms_pix.npy')
    # path2 = os.path.join(os.path.dirname(__file__), 'dcms_id.npy')
    # x_array = np.load(path1)
    # print('Data array loaded.')
    # print(x_array.shape)
    # id_array = np.load(path2)
    # print('ID array loaded.')

    # If available, upload data (issue with gdcm and MacOs).
    # Otherwise run the generate_image_and_id_arrays(path) function
    # x_array = np.load('src/util/dcms_pix.npy')
    # print('Data array loaded.')
    # id_array = np.load('src/util/dcms_id.npy')
    # print('ID array loaded.')


    # Performing PCA
    n_components = 3
    pca(n_components=n_components)
    pca = PCA(.95) # It means that scikit-learn choose the minimum number of principal components
    print('PCA model created! Starting training...')
    # such that 95% of the variance is retained.

    x_train = PCA_function(x_array, pca)
    print('Training complete! Let us now save the model as a pickle file...')

    # These things are computationally expensive, so save your models!
    filename = 'pca_'+str(pca.n_components)+'PC_'+str(time.strftime("%d_%m_%Y"))+'_'+str(time.strftime("%H:%M:%S"))+'.pickle'
    pickle.dump(pca, open(filename, 'wb'))
    print('Model saved!')

    # Create a dataframe for all the data:
    columns = []
    for i in range(1, n_components+1):
        string = 'principal component '+ str(i)
        columns.append(string)

    print('Creating a dataframe to plot the data in 3D...')
    principalDf = pd.DataFrame(data=x_train, columns=columns)

    print("Let's plot!")
    static_plotting(principalDf, id_array)
    # interactive_plotting(principalDf, id_array)
    print("Aaaand, we're done!")

    # # Trying to reconstruct the image
    # # Thanks to: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb
    # approximation = pca.inverse_transform(x_train)
    #
    #
    # #######################################
    # # WORKING ON PCA COMPUTED ON LEONHARD #
    # #######################################
    # filename = 'pca_0.95PC_13_01_2019_18:19:42.pickle'
    # pca_model = pickle.load(open('src/util/'+filename, 'rb'))
    # n_components = pca_model.n_components_
    #
    # # Perform PCA transformation on data
    # x_reshaped = x_array.reshape(x_array.shape[0], 512*512)
    # x_transformed = pca_model.transform(x_reshaped)
    #
    # # Create a dataframe for all the data:
    # columns = []
    # for i in range(1, n_components+1):
    #     string = 'principal component '+ str(i)
    #     columns.append(string)
    #
    # principalDf = pd.DataFrame(data=x_transformed, columns=columns)
    #
    # # Plotting the transformed data:
    # static_plotting(principalDf, id_array)
    # interactive_plotting(principalDf, id_array)
    #
    # # Creating approximations with all components:
    # approximation = pca_model.inverse_transform(x_transformed)
    # slice_approx = approximation[1235, :]  # pick a slice to plot
    # approx_reshaped = slice_approx.reshape(512, 512)  # reshape so you get back your image :)
    # greyscale_plot(approx_reshaped)
    #
    # # Trying to isolate single PC:
    # # https://stackoverflow.com/questions/32750915/pca-inverse-transform-manually
    #
    # # # Isolating PC1:
    # PC1 = pca_model.components_[0,:]
    # # U_reduced = np.zeros(pca_model.components_.shape)  # Creating a new U matrix which will have only one PC
    # # U_reduced[0,:] = PC1  # here we let U_reduced have only PC1
    # # PC1_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_  # recreating approximation with only PC1
    # # # Save your arrays and spare your computer :)
    # # # np.save('src/util/PC1_approximation.npy', PC1_approx)
    # # greyscale_plot(PC1_approx[20,:].reshape(512, 512))  # resembles an x-ray
    # #
    # # # Isolating PC2:
    # PC2 = pca_model.components_[1,:]
    # # U_reduced = np.zeros(pca_model.components_.shape)
    # # U_reduced[1,:]=PC2
    # # PC2_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # # # Save your arrays and spare your computer :)
    # # # np.save('src/util/PC2_approximation.npy', PC2_approx)
    # # greyscale_plot(PC2_approx[120,:].reshape(512, 512))  # maybe encoding pixel values?
    # #
    # # # Isolating PC3:
    # PC3 = pca_model.components_[2,:]
    # # U_reduced = np.zeros(pca_model.components_.shape)
    # # U_reduced[2,:]=PC3
    # # PC3_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # # # Save your arrays and spare your computer :)
    # # # np.save('src/util/PC3_approximation.npy', PC3_approx)
    # # greyscale_plot(PC3_approx[1209,:].reshape(512, 512))
    # #
    # # # Isolating PC4:
    # PC4 = pca_model.components_[3,:]
    # # U_reduced = np.zeros(pca_model.components_.shape)
    # # U_reduced[3,:] = PC4
    # # PC4_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # # # Save your arrays and spare your computer :)
    # # # np.save('src/util/PC3_approximation.npy', PC3_approx)
    # # greyscale_plot(PC4_approx[1209,:].reshape(512, 512))
    #
    # # Now let's combine PC!
    # U_reduced = np.zeros(pca_model.components_.shape)
    # U_reduced[0,:] = PC1
    # U_reduced[1,:] = PC2
    # PC12_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # greyscale_plot(PC12_approx[1326,:].reshape(512, 512))
    #
    # U_reduced = np.zeros(pca_model.components_.shape)
    # U_reduced[0,:] = PC1
    # U_reduced[1,:] = PC2
    # U_reduced[2,:] = PC3
    # PC123_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # greyscale_plot(PC123_approx[1269,:].reshape(512, 512))
    #
    # # Removing components:
    # U_reduced = pca_model.components_
    # U_reduced[1,:] = 0
    # PC_less1_approx = np.dot(x_transformed, U_reduced) + pca_model.mean_
    # greyscale_plot(PC_less1_approx[1235,:].reshape(512, 512))
