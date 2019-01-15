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

    return files


def _push_to_positive_domain(dcm_file, neg_min, filename):
    dcm_file.SliceLocation += abs(neg_min)

    #dcm_file.save_as(filename)
    return dcm_file
    #print('with new location {}'.format(dcm_file.SliceLocation))

def _normalize(dcm_file, max, filename):
    #print(max)
    if max != 0:
        print('divide {} by {}'.format(dcm_file.SliceLocation, max))
        dcm_file.SliceLocation = (dcm_file.SliceLocation/max)
        return dcm_file
        #print('normalized value {}'.format(dcm_file.SliceLocation))
        #dcm_file.save_as(filename)
    else:
        return dcm_file


def main(path):
    files = _input_files(path)
    max_z = [0, '']
    min_z = [1000, '']
    locations = []
    compare_z = []
    subj_extremes = defaultdict(lambda : defaultdict(int))

    # loop through all subject studies
    for f in files:
        #print(f)
        subj_extremes[f]['min'] = 10000
        subj_extremes[f]['max'] = -10000
        # get min and max z-slice value
        for i in files[f]['files']:
            #print('working on {}'.format(i))
            ds = pydicom.dcmread(i)               
            location = ds.get('SliceLocation', "(missing)")
            location = float(location)
            #print('slice location {}'.format(location))
            if location > max_z[0]:
                max_z = [location, i]
            if location < min_z[0]:
                min_z = [location,i]
    
            if location < subj_extremes[f]['min']:
                #print('{} is smaller than {}'.format(location,subj_extremes[f]['min']))
                subj_extremes[f]['min'] = location

            if location > subj_extremes[f]['max']:
                subj_extremes[f]['max'] = location
        #print(subj_extremes[f]['min'])
        #print(subj_extremes[f]['max'])
        #print('batch extremes: min: {}, max:{}'.format(subj_extremes[f]['min'],subj_extremes[f]['max']))

        # if minimal value negative
        if float(subj_extremes[f]['min']) < 0:
            #print('replace values for {}'.format(files[f]['files']))
            for j in files[f]['files']:
                ds2 = pydicom.dcmread(j)
                location = ds.get('SliceLocation', "(missing)")
                #print('replace old value {}...'.format(ds2.SliceLocation))
                new_ds = _push_to_positive_domain(ds2, subj_extremes[f]['min'], j)
                #print(new_ds.SliceLocation)
                new_ds.save_as(j)

    for f in files:
        subj_extremes[f]['min'] = 10000
        subj_extremes[f]['max'] = 0
        for l in files[f]['files']:
            #print(i)
            ds = pydicom.dcmread(l)               
            location = ds.get('SliceLocation', "(missing)")
            if location > max_z[0]:
                max_z = [location, l]
            if location < min_z[0]:
                min_z = [location,l]
    
            if location < subj_extremes[f]['min']:
                #print('{} is smaller than {}'.format(location,subj_extremes[f]['min']))
                subj_extremes[f]['min'] = location

            if location > subj_extremes[f]['max']:
                #print('{} is larger than {}'.format(location,subj_extremes[f]['max']))
                subj_extremes[f]['max'] = location

        for k in files[f]['files']:
            ds3 = pydicom.dcmread(k)
            print('replace old value {}...'.format(ds3.SliceLocation))
            new_ds = _normalize(ds3, subj_extremes[f]['max'], k)
            print(new_ds.SliceLocation)
            new_ds.save_as(k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform dcm \
        dataset into TFRecord format')

    parser.add_argument('-p', '--path',
                        help='Path where to start searching', default='.')

    args = parser.parse_args()

    main(args.path)