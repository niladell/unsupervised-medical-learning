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


def _rewrite_dcm(dcm_file, neg_min, filename):
    dcm_file.SliceLocation += abs(neg_min)
    dcm_file.save_as(filename)
    print('rewrote {}'.format(filename))


def main(path, output_file):
    files = _input_files(path)
    max_z = [0, '']
    min_z = [1000, '']
    locations = []
    compare_z = []
    subj_extremes = defaultdict(lambda : defaultdict(int))

    # loop through all subject studies
    for f in files:
        # get min and max z-slice value
        for i in files[f]['files']:
            ds = pydicom.dcmread(i)               
            location = ds.get('SliceLocation', "(missing)")
            if location > max_z[0]:
                max_z = [location, i]
            if location < min_z[0]:
                min_z = [location,i]
    
            if location < subj_extremes[f]['min']:
                subj_extremes[f]['min'] = location

            if location > subj_extremes[f]['max']:
                subj_extremes[f]['max'] = location

        # if minimal value negative
        if int(subj_extremes[f]['min']) < 0:
            for i in files[f]['files']:
                _rewrite_dcm(ds, subj_extremes[f]['min'], i)


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
    main(args.path, args.output)
    #visualize(array)