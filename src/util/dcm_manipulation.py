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
plotly.tools.set_credentials_file(username='InesPereira', api_key='aXsds6rXqJPaaFnw9FFf')
import pickle
import time


def generate_image_and_id_arrays(list_of_paths):
    # Loop to create a list with the pixel data per slice
    x = []
    id = []
    # for file in tqdm((list_of_paths)):
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


def static_plotting(principalDf, id_array, title = 'PCA of 20 subjects with no lesions'):
    # Defining colors automatically:
    colors = define_colors(len(set(id_array)))

    # Plotting
    fig = plt.figure(figsize = (8,8))
    # ax = fig.add_subplot(1,1,1)
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
            x=principalDf.loc[idx, 'principal component 168'],
            y=principalDf.loc[idx, 'principal component 68'],
            z=principalDf.loc[idx, 'principal component 52'],
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
            xaxis=XAxis(title='Principal Component 168'),
            yaxis=YAxis(title='Principal Component 68'),
            zaxis=ZAxis(title='Principal Component 52')
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename= plot_name)


def main(x_array, id_array):
    # Performing PCA
    n_components = 3
    # pca(n_components=n_components)
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
    interactive_plotting(principalDf, id_array, x_array)
    print("Aaaand, we're done!")
# Another library for interactive plots: https://bokeh.pydata.org/en/latest/docs/gallery/iris_splom.html


if __name__ == "__main__":

    # YOU HAVE TO BE IN THE DATASET FOLDER TO RUN THIS SCRIPT

    # PCA on healthy subjects

    path1 = os.path.join(os.path.dirname(__file__), 'x_healthy_array.npy')
    path1= 'src/util/x_healthy_array.npy'
    path2 = os.path.join(os.path.dirname(__file__), 'id_healthy_array.npy')
    path2 = 'src/util/id_healthy_array.npy'
    x_array_healthy = np.load(path1)
    print('Data array loaded.')
    print(x_array_healthy.shape)
    id_array_healthy = np.load(path2)
    print('ID array loaded.')

    # main(x_array_healthy, id_array_healthy)


    # PCA on healthy and hemorrhages:

    path1 = os.path.join(os.path.dirname(__file__), 'x_hemorrhage_array.npy')
    path1 = 'src/util/x_hemorrhage_array.npy'
    path2 = os.path.join(os.path.dirname(__file__), 'id_hemorrhage_array.npy')
    path2 = 'src/util/id_hemorrhage_array.npy'
    x_hemorrhage_array = np.load(path1)
    x_hh_array = np.append(x_array_healthy, x_hemorrhage_array, axis=0)
    print('Data array loaded.')
    print(x_hh_array.shape)
    id_hemorrhage_array = np.load(path2)
    id_hh_array = np.append(id_array_healthy, id_hemorrhage_array, axis=0)
    print('ID array loaded.')

    # main(x_hh_array, id_hh_array)


    # PCA on healthy and fractures:

    path1 = os.path.join(os.path.dirname(__file__), 'x_frac_array.npy')
    path1 = 'src/util/x_frac_array.npy'
    path2 = os.path.join(os.path.dirname(__file__), 'id_frac_array.npy')
    path2 = 'src/util/id_frac_array.npy'
    x_frac_array = np.load(path1)
    x_hf_array = np.append(x_array_healthy, x_frac_array, axis=0)
    print('Data array loaded.')
    print(x_hf_array.shape)
    id_frac_array = np.load(path2)
    id_hf_array = np.append(id_array_healthy, id_frac_array, axis=0)
    print('ID array loaded.')

    # main(x_hf_array, id_hf_array)


    # PCA on all of them!
    print("Let's reduce the number of data points :)")
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
    print("Let's reduce the number of data points :)")
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
    print("We have, in total, for this PCA, "+ str(len(set(id_hhf30_array2))) + " different subjects.")


    # Creating the labels for the training and validation datasets
    labels1_hh30 = np.append(id_healthy, id_frac, axis=0)
    labels1_hh30 = np.append(labels1_hh30, id_hemorrhage, axis=0)

    labels2_hh30 = np.append(id_healthy2, id_frac2, axis=0)
    labels2_hh30 = np.append(labels2_hh30, id_hemorrhage2, axis=0)



    # #######################################
    # # WORKING ON PCA COMPUTED ON LEONHARD #
    # #######################################
    filename = '/Users/ines/Downloads/das_pca.pickle'
    pca_model = pickle.load(open(filename, 'rb'))
    n_components = pca_model.n_components_

    # Load the data
    x_array = np.load('src/util/x_healthy_array.npy')
    print('Data array loaded.')
    id_array = np.load('src/util/id_healthy_array.npy')
    print('ID array loaded.')

    # Perform PCA transformation on data
    x_transformed = pca_model.transform(x_hhf30_array2.reshape(x_hhf30_array2.shape[0], 512*512))

    # Create a dataframe for all the data:
    columns = []
    for i in range(1, n_components+1):
        string = 'principal component '+ str(i)
        columns.append(string)

    principalDf = pd.DataFrame(data=x_transformed, columns=columns)

    # Plotting the transformed data:
    static_plotting(principalDf, labels2_hh30, title='')
    interactive_plotting(principalDf, labels2_hh30, 'PCA hhf30 on h, h and f, PC168, 68, 52')

    for i in range(15):
        greyscale_plot(pca_model.components_[i,:].reshape(512,512))

    plt.loglog(pca_model.explained_variance_ratio_)
    plt.plot(pca_model.explained_variance_ratio_)
    plt.show()

    ##########################
    # Training models on top #
    ##########################

    # Training an SVM

    from sklearn.svm import SVC
    clf = SVC(C=10, gamma='auto', kernel = 'linear')
    clf.fit(x_transformed, labels2_hh30)

    # Let's see how much it scores with the other half of the data set:
    test_transformed = pca_model.transform(x_hhf30_array.reshape(x_hhf30_array.shape[0], 512*512))
    clf.score(test_transformed, labels1_hh30)
    clf.score(x_transformed, labels2_hh30)

    clf.get_params()
    a = clf._get_coef()

    plt.figure()
    plt.plot(clf._get_coef())
    plt.show()


    def interactive_plotting(principalDf, id_array, plot_name):
        # Interactive plotting with plotly:
        colors = define_colors(len(set(id_array)))
        set_ids = set(id_array)
        data_full = []
        for iden, color in zip(set_ids, colors):
            idx = id_array == iden
            trace1 = go.Bar(
            x=list(range(346)),
            y=principalDf.loc[idx, 'principal component 52'],
            name=iden
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
                xaxis=XAxis(title='Principal Component 168'),
                yaxis=YAxis(title='Principal Component 68'),
                zaxis=ZAxis(title='Principal Component 52')
            )
        )
        fig = go.Figure(data=data)
        py.iplot(fig, filename='test comp 52')


    tr = []
    for i in range(3):
        # plt.figure()
        # plt.bar(range(346),clf.coef_[i,:]**2)
        # plt.show()

        tr.append(go.Bar(
            x=list(range(346)),
            y=clf.coef_[i, :] ** 2,
            name=i
        ))

        for iden, color in zip(set_ids, colors):
            idx = id_array == iden
            trace1 = go.Scatter3d(
                x=principalDf.loc[idx, 'principal component 168'],
                y=principalDf.loc[idx, 'principal component 68'],
                z=principalDf.loc[idx, 'principal component 52'],
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

    py.iplot(go.Figure(tr), filename='hey there cofficients')

    greyscale_plot(pca_model.components_[52, :].reshape(512, 512))


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