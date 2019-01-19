from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import re
import argparse
from collections import defaultdict, OrderedDict
import os
import itertools

""" 


"""
def get_dcms(path):
    list_of_dcm = defaultdict(lambda : defaultdict(list))
    for dirpath, dirname, filenames in os.walk(path):
        for file in filenames:
            pattern = re.compile(r'.dcm$')
            m = re.search(pattern, file)
            if m is not None:
                dcm_path = dirpath + '/' + file
                list_of_dcm[dcm_path[:-13]]['files'].append(dcm_path)
    return list_of_dcm

def _push_to_positive_domain(dcm_file, neg_min, filename):
    dcm_file.SliceLocation += abs(neg_min)

    #dcm_file.save_as(filename)
    return dcm_file
    #print('with new location {}'.format(dcm_file.SliceLocation))

def _normalize(dcm_file, max, filename):
    #print(max)
    if max != 0:
        #print('divide {} by {}'.format(dcm_file.SliceLocation, max))
        dcm_file.SliceLocation = (dcm_file.SliceLocation/max)
        return dcm_file
        #print('normalized value {}'.format(dcm_file.SliceLocation))
        #dcm_file.save_as(filename)
    else:
        return dcm_file


def main(path):
    files = get_dcms(path)
    max_z = [0, '']
    min_z = [1000, '']
    locations = []
    compare_z = []
    subj_extremes = defaultdict(lambda : defaultdict(int))

    # loop through all subject studies
    for f in files:

        subj_extremes[f]['min'] = 10000
        subj_extremes[f]['max'] = -10000

        for i in files[f]['files']:
            #print('working on {}'.format(i))
            ds = pydicom.dcmread(i)               
            location = ds.get('SliceLocation', "(missing)")
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
            #print('replace old value {}...'.format(ds3.SliceLocation))
            new_ds = _normalize(ds3, subj_extremes[f]['max'], k)
            #print(new_ds.SliceLocation)
            new_ds.save_as(k)
    _check(files)

def _check(files):
        subj = {}
        images = []
        for f in files:
            for i in files[f]['files']:
                ds = pydicom.dcmread(i)
                location = ds.get('SliceLocation', "(missing)")
                subj[location] = i
        od = OrderedDict(sorted(subj.items()))
        #for k,v in od.items():
            #print('{}\t{}'.format(k,v))
        items = list(od.items())
        a = items[0:3]
        b = items[-4:-1]
        imag_indexes = itertools.chain(a, b)
        for k,v in imag_indexes:
            ds = pydicom.dcmread(v)
            img_raw = ds.pixel_array
            images.append([img_raw, ds.SliceLocation])
        image_array = np.asarray(images)
        visualize(image_array)

            
def visualize(image_array):
    w=512
    h=512
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2
    for i in range(1, columns*rows +1):
        img = image_array[i-1][0]
        a = fig.add_subplot(rows, columns, i)
        a.set_title("location "+str(image_array[i-1][1])[0:8])  # set title
        plt.imshow(img)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform dcm \
        dataset into TFRecord format')

    parser.add_argument('-p', '--path',
                        help='Path where to start searching', default='.')

    args = parser.parse_args()

    (main(args.path))

    