'''
Script to run PCAs on selected subjects of the CQ500 dataset.
You have to be in the same folder as those subject folders to run this script.
To actually run this script, you will need to comment in the calls for main function.
To plot with plotly, you will need to create your own credentials.

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
plotly.tools.set_credentials_file(username='InesPereira', api_key='aXsds6rXqJPaaFnw9FFf')
import pickle
import time


def generate_image_and_id_arrays(list_of_paths):
    # Loop to create a list with the pixel data per slice
    x = []
    id = []
    for file in tqdm((list_of_paths)):
        pattern = re.compile(r'.dcm$')
        m = re.search(pattern, file)
        if m is not None:
            # print(file)
            dcm = pydicom.dcmread(file)
            identity = dcm.get('PatientID')
            if dcm.pixel_array.shape == (512, 512):
                x.append(dcm.pixel_array)
                id.append(identity)

    x_array = np.asarray(x)  # convert to np.array for ease of manipulation
    id_array = np.asarray(id)

    return x_array, id_array

def PCA_function(x_array, pca):
    print('Starting PCA on the given data...')
    # Preprocessing:
    x_reshaped = x_array.reshape(x_array.shape[0], 512*512)  # Reshape so you can run sk-learn pca

    # Normalization (recommended)
    sc = StandardScaler()
    x_reshaped = sc.fit_transform(x_reshaped)
    print('Normalization of the data done!')

    # Check out how many components are necessary to explain the above defined variance:
    # pca.n_components_

    x_train = pca.fit_transform(x_reshaped)
    print('Amazing! Your PCA has been completed!')

    return x_train


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


def static_plotting(principalDf, id_array, title = ''):
    # Defining colors automatically:
    colors = define_colors(len(set(id_array)))

    # Plotting
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15, labelpad=15)
    ax.set_ylabel('Principal Component 2', fontsize = 15, labelpad=15)
    ax.set_zlabel('Principal Component 3', fontsize = 15, labelpad=15)
    ax.set_title(title, fontsize = 18)
    set_ids = set(id_array)
    for iden, color in zip(set_ids, colors):
        idx = id_array == iden
        ax.scatter(principalDf.loc[idx, 'principal component 1'],
                   principalDf.loc[idx, 'principal component 2'],
                   principalDf.loc[idx, 'principal component 3'],
                   cmap=color,
                   )
    ax.legend(set_ids, loc= 'upper left')
    ax.grid()
    plt.show()


def interactive_plotting(principalDf, id_array, plot_name):
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
    py.iplot(fig, filename= plot_name)


def load_data_and_ids(x_array_path, id_array_path):
    x_array = np.load(x_array_path)
    print('Data array loaded.')
    print(x_array.shape)
    id_array = np.load(id_array_path)
    print('ID array loaded.')
    print(id_array.shape)
    return x_array, id_array


def main(x_array, id_array):
    # Performing PCA
    n_components = 3
    pca = PCA(.95)
    print('PCA model created! Starting training...')
    # such that 95% of the variance is retained.

    x_train = PCA_function(x_array, pca)
    print('Training complete! Let us now save the model as a pickle file...')

    # These models can be computationally expensive, so we're saving them:
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
    interactive_plotting(principalDf, id_array, x_array)
    print("Aaaand, we're done!")
# Another library for interactive plots: https://bokeh.pydata.org/en/latest/docs/gallery/iris_splom.html




if __name__ == "__main__":


    # PCA on subjects with no lesions
    x_array_healthy, id_array_healthy = load_data_and_ids('src/util/x_healthy_array.npy', 'src/util/id_healthy_array.npy')
    # main(x_array_healthy, id_array_healthy)


    # PCA on healthy and hemorrhages:
    x_hemorrhage_array, id_hemorrhage_array = load_data_and_ids('src/util/x_hemorrhage_array.npy', 'src/util/id_hemorrhage_array.npy')

    x_hh_array = np.append(x_array_healthy, x_hemorrhage_array, axis=0)
    print('Data arrays of subjects with no lesion and with hematomas/hemorrhage appended.')
    print(x_hh_array.shape)
    id_hh_array = np.append(id_array_healthy, id_hemorrhage_array, axis=0)
    print('ID arrays also appended.')

    # main(x_hh_array, id_hh_array)


    # PCA on healthy and fractures:
    x_frac_array, id_frac_array = load_data_and_ids('src/util/x_frac_array.npy', 'src/util/id_frac_array.npy')

    x_hf_array = np.append(x_array_healthy, x_frac_array, axis=0)
    print('Data arrays of subjects with no lesion and with fractures appended.')
    print(x_hf_array.shape)
    id_hf_array = np.append(id_array_healthy, id_frac_array, axis=0)
    print('ID arrays also appended.')

    # main(x_hf_array, id_hf_array)


    # PCA on all of them!
    print("Let's reduce the number of data points...")
    print('Let us start with healthy!')
    x_array_healthy_half = x_array_healthy[:2250, :,:]
    print(x_array_healthy_half.shape)
    id_array_healthy_half = id_array_healthy[:2250]
    print(id_array_healthy_half.shape)

    id_healthy = np.full(id_array_healthy_half.shape, 'No lesions')

    print("We have " + str(len(set(id_array_healthy_half)))+" subjects with no lesions.")


    print("Now the bleeders")
    x_hemorrhage_array_half = x_hemorrhage_array[:2500, :,:]
    print(x_hemorrhage_array_half.shape)
    id_hemorrhage_array_half = id_hemorrhage_array[:2500]
    print(id_hemorrhage_array_half.shape)

    id_hemorrhage = np.full(id_hemorrhage_array_half.shape, 'Hemorrhage')

    print("We have " + str(len(set(id_hemorrhage_array_half))) + " subjects with hemorrhages or hematomas.")

    print("And now the broken.")
    x_frac_array_half = x_frac_array[:2750, :,:]
    print(x_frac_array_half.shape)
    id_frac_array_half = id_frac_array[:2750]

    id_frac = np.full(id_frac_array_half.shape, 'Fracture')

    print(id_frac_array_half.shape)
    print("We have " + str(len(set(id_frac_array_half))) + " subjects with a fracture.")

    x_hhf30_array = np.append(x_array_healthy_half, x_frac_array_half, axis=0)
    x_hhf30_array = np.append(x_hhf30_array, x_hemorrhage_array_half, axis=0)

    id_hhf30_array = np.append(id_array_healthy_half, id_frac_array_half, axis=0)
    id_hhf30_array = np.append(id_hhf30_array, id_hemorrhage_array_half, axis=0)
    print("We have, in total, for this PCA, "+ str(len(set(id_hhf30_array))) + " different subjects.")

    # main(x_hhf30_array, id_hhf30_array)


    #################################################################

    # Getting the validation datasets:
    print("Let's get ourselves some validation datasets")
    print('Let us start with healthy!')
    x_array_healthy_half2 = x_array_healthy[2250:, :,:]
    print(x_array_healthy_half2.shape)
    id_array_healthy_half2 = id_array_healthy[2250:]

    id_healthy2 = np.full(id_array_healthy_half2.shape, 'No lesions')

    print(id_array_healthy_half.shape)
    print("We have " + str(len(set(id_array_healthy_half)))+" subjects with no lesions.")


    print("Now the bleeders")
    x_hemorrhage_array_half2 = x_hemorrhage_array[2500:, :,:]
    print(x_hemorrhage_array_half2.shape)
    id_hemorrhage_array_half2 = id_hemorrhage_array[2500:]

    id_hemorrhage2 = np.full(id_hemorrhage_array_half2.shape, 'Hemorrhage')

    print(id_hemorrhage_array_half2.shape)
    print("We have " + str(len(set(id_hemorrhage_array_half2))) + " subjects with hemorrhages or hematomas.")

    print("And now the broken.")
    x_frac_array_half2 = x_frac_array[2750:, :,:]
    print(x_frac_array_half2.shape)
    id_frac_array_half2 = id_frac_array[2750:]

    id_frac2 = np.full(id_frac_array_half2.shape, 'Fracture')

    print(id_frac_array_half2.shape)
    print("We have " + str(len(set(id_frac_array_half2))) + " subjects with a fracture.")

    x_hhf30_array2 = np.append(x_array_healthy_half2, x_frac_array_half2, axis=0)
    x_hhf30_array2 = np.append(x_hhf30_array2, x_hemorrhage_array_half2, axis=0)

    id_hhf30_array2 = np.append(id_array_healthy_half2, id_frac_array_half2, axis=0)
    id_hhf30_array2 = np.append(id_hhf30_array2, id_hemorrhage_array_half2, axis=0)
    print("We have, in total, for this valiadtion dataset, "+ str(len(set(id_hhf30_array2))) + " different subjects.")


    # Creating the labels for the training and validation datasets
    labels1_hh30 = np.append(id_healthy, id_frac, axis=0)
    labels1_hh30 = np.append(labels1_hh30, id_hemorrhage, axis=0)

    labels2_hh30 = np.append(id_healthy2, id_frac2, axis=0)
    labels2_hh30 = np.append(labels2_hh30, id_hemorrhage2, axis=0)


    #########################################
    #    WORKING ON A PRE-COMPUTED PCA      #
    #########################################

    filename = '/Users/ines/Downloads/das_pca.pickle'  # path where your PCA is stored.
    pca_model = pickle.load(open(filename, 'rb'))

    # Perform PCA transformation on data
    x_transformed = pca_model.transform(x_hhf30_array2.reshape(x_hhf30_array2.shape[0], 512*512))

    # Create a dataframe for all the data:
    columns = []
    for i in range(1, pca_model.n_components_+1):
        string = 'principal component '+ str(i)
        columns.append(string)

    principalDf = pd.DataFrame(data=x_transformed, columns=columns)

    # Plotting the transformed data:
    static_plotting(principalDf, labels2_hh30, title='')
    interactive_plotting(principalDf, labels2_hh30, 'PCA hhf30')

    for i in range(15):
        greyscale_plot(pca_model.components_[i,:].reshape(512,512))

    plt.loglog(pca_model.explained_variance_ratio_)
    plt.plot(pca_model.explained_variance_ratio_)
    plt.show()

    ############################################
    # Training models on top of projected data #
    ############################################

    # Training an SVM to see if we can extract clinically relevant features.

    from sklearn.svm import SVC
    clf = SVC(C=1, gamma='auto', kernel = 'rbf')
    clf.fit(x_transformed, labels2_hh30)
    print("SVM fitted!")

    # Let's see how much it scores with the other half of the data set:
    test_transformed = pca_model.transform(x_hhf30_array.reshape(x_hhf30_array.shape[0], 512*512))
    print("Validation set transformed! Now let's compute some score.")
    print("On the training set, we've reached a classification accuracy of "+ str(clf.score(x_transformed, labels2_hh30)))
    print("On the test set, we've reached a classification accuracy of "+ str(clf.score(test_transformed, labels1_hh30)))

