from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import re
import argparse
from collections import defaultdict

""" 
1. normalize: for every subj_study, push z-loc boundaries. 
-> was ist unique beim filename? 

"""

def _input_files(path):
    """Check if z-location is consistent
    Note: Run in directory where dcms are

    Args:
        path (str): root path where to start searching
        filename_pattern (str): Any file with this substring will be
            selected

    Returns:
        (list): List of all files found (with the path from root)
    """
    files = defaultdict(lambda : defaultdict(list))
    for p, d, folder in tf.gfile.Walk(path):
        #print(' Folder walk {}, {}, {}'.format(p, d, folder))
        for f in folder:
            if '.dcm' in f:
                #print(files[f[:-13]]['files'])
                files[f[:-13]]['files'].append(f)

                #files.append(f)
    #print(files['CQ500CT13_CT_PRE_CONTRAST_THIN']['files'])
    return files
    # group files to

def load_all_data_sameloc(path, output_file):
    files = _input_files(path)
    #f = open(output_file, "a")
    max_z = [0, '']
    min_z = [1000, '']
    locations = []
    compare_z = []
    images = []
    subj_extremes = defaultdict(lambda : defaultdict(int))

    # loop through all subject studies
    for f in files:
        # for each dcm file
        for i in files[f]['files']:               
            ds = pydicom.dcmread(i)
            identity = ds.get('PatientID')
            identity_nr = re.findall(r'\d+$', identity)[0]
            if int(identity_nr):
            #if int(identity_nr) < 50:

                location = ds.get('SliceLocation', "(missing)")
                if location > max_z[0]:
                    max_z = [location, i]
                if location < min_z[0]:
                    min_z = [location,i]
    
    
                if location < subj_extremes[f]['min']:
                    subj_extremes[f]['min'] = location
    
                if location > subj_extremes[f]['max']:
                    subj_extremes[f]['max'] = location
                #print(location)
                #if location:
                if location >= 0 and location <= 20:
                    #compare_z.append(i)
                    #print('{} between defined range'.format(i))
                    img_raw = ds.pixel_array
                    #print(img_raw.shape)
                    images.append([img_raw, identity_nr])
                    # store all images + filename

    #for k in subj_extremes:
        #print('subj:{},min:{},max:{}'.format(k, subj_extremes[k]['min'], subj_extremes[k]['max']))
    image_array = np.asarray(images)
    return image_array
    
        #for j in compare_z:
        #    f.write(j)
    
        #print(max_z[0], min_z[0])
    
        #print(image_array.shape)

def load_all_data_samesubj(path, output_file):
    files = _input_files(path)
    f = open(output_file, "a")
    max_z = [0, '']
    min_z = [1000, '']
    compare_z = []
    images = []

    for f in files:
        # for each dcm file
        for i in files[f]['files']:
            if i.endswith('00000.dcm'):
                ds = pydicom.dcmread(i)
                identity = ds.get('PatientID')
                identity_nr = re.findall(r'\d+$', identity)[0]
                img_raw = ds.pixel_array
                images.append([img_raw, identity_nr])
    image_array = np.asarray(images)
    return image_array


def visualize(image_array):
    randimags = np.random.choice(image_array.shape[0], 20, replace=False)
    #print(randimags)
    w=512
    h=512
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    counter = 0
    for i in range(1, columns*rows +1):
        index = randimags[counter]
        img = image_array[index][0]
        a = fig.add_subplot(rows, columns, i)
        a.set_title("ID "+str(image_array[index][1]))  # set title
        plt.imshow(img)
        counter += 1
    plt.tight_layout()
    plt.show()
    # print filenames of random images
    #for elem in randimags:
        #print(image_array[elem][1])
    #for i in randimags:
        #print(files[i])
    #f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform dcm \
        dataset into TFRecord format')
    parser.add_argument('-o', '--output',
                        help='Output TFRecord file', required=True)
    parser.add_argument('-p', '--path',
                        help='Path where to start searching', default='.')
    #parser.add_argument('-i', '--filename',
                        #help='File pattern to look for', required=True)
    args = parser.parse_args()

    #_input_files(args.path)
    array = load_all_data_sameloc(args.path, args.output)
    visualize(array)