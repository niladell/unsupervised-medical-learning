#!/usr/bin/env python3

""" 
This script processes a bunch of .dcm files, squishes their slice location values
to [0,1], normalizes and adjusts the pixel values to an appropriate windowing and 
splis the dataset into 3 batches, organizing slices at the bottom, middle and top
respectively. The batches are dictionaries with study sessions as keys and the 
corresponding pixel arrays as values. Eventually, the three batches are stored
separately in pickle format.
"""

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
import cv2 as cv
import pickle

list_forbidden_folders = ['CT 4cc sec 150cc D3D on',
                          'CT 4cc sec 150cc D3D on-2',
                          'CT 4cc sec 150cc D3D on-3',
                          'CT POST CONTRAST',
                          'CT POST CONTRAST-2',
                          'CT BONE',
                          'CT I To S',
                          'CT PRE CONTRAST BONE',
                          'CT Thin Bone',
                          'CT Thin Stnd',
                          'CT 0.625mm',
                          'CT 0.625mm-2',
                          'CT 5mm POST CONTRAST',
                          'CT ORAL IV',
                          'CT 55mm Contrast',
                          'CT BONE THIN',
                          'CT 3.753.75mm Plain',
                          'CT Thin Details',
                          'CT Thin Stand']

def get_dcms(path, pattern=re.compile(r'.dcm')):
    list_of_dcm = defaultdict(lambda : defaultdict(list))
    for dirpath, dirname, filenames in os.walk(path):
        for elem in list_forbidden_folders:
            if elem not in dirpath:
                for file in filenames:
                    pattern = pattern
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

def _def_0_start(dcm_file, pos_min, filename):
    dcm_file.SliceLocation -= abs(pos_min)
    return dcm_file


def _normalize(dcm_file, max, filename):
    """
    squishes [0, x] -> [0,1]
    """
    if max != 0:
        #print('divide {} by {}'.format(dcm_file.SliceLocation, max))
        dcm_file.SliceLocation = (dcm_file.SliceLocation/max)
        return dcm_file
        #print('normalized value {}'.format(dcm_file.SliceLocation))
        #dcm_file.save_as(filename)
    else:
        return dcm_file

def renormalize_z(files):
    """ 
    args: dict with study batch as key, filenames of dcm files as values

    directly overrides the dcm files

    squishes values of slice locations to [0, 1] and rescales pixel values
    """
    subj_extremes = defaultdict(lambda : defaultdict(int))

    # loop through all subject studies
    print('start renormalization of slice location')
    for f in files:

        subj_extremes[f]['min'] = 10000
        subj_extremes[f]['max'] = -10000

        for i in files[f]['files']:
            #print('working on normalization of {}'.format(i))
            ds = pydicom.dcmread(i)               
            location = ds.get('SliceLocation', "(missing)")   
            if location < subj_extremes[f]['min']:
                #print('{} is smaller than {}'.format(location,subj_extremes[f]['min']))
                subj_extremes[f]['min'] = location

            if location > subj_extremes[f]['max']:
                subj_extremes[f]['max'] = location

        # push lowest slicelocation value to 0
        if float(subj_extremes[f]['min']) < 0:
            #print('replace values for {}'.format(files[f]['files']))
            for j in files[f]['files']:
                ds2 = pydicom.dcmread(j)
                location = ds.get('SliceLocation', "(missing)")
                #print('replace old value {}...'.format(ds2.SliceLocation))
                new_ds = _push_to_positive_domain(ds2, subj_extremes[f]['min'], j)
                new_ds.save_as(j)

        # pull lowest slicelocation value to 0
        elif float(subj_extremes[f]['min']) > 0:
            for j in files[f]['files']:
                ds2 = pydicom.dcmread(j)
                location = ds.get('SliceLocation', "(missing)")
                #print('replace old value {}...'.format(ds2.SliceLocation))
                new_ds = _def_0_start(ds2, subj_extremes[f]['min'], j)
                new_ds.save_as(j)
        #print('pushed {} to positive domain with 0 starting point'.format(f))

    # get new min and max values (after defining new center)
    print('finished setting new 0')
    for f in files:
        subj_extremes[f]['min'] = 10000
        subj_extremes[f]['max'] = 0
        for l in files[f]['files']:
            #print(i)
            ds = pydicom.dcmread(l)               
            location = ds.get('SliceLocation', "(missing)")

            if location < subj_extremes[f]['min']:
                #print('{} is smaller than {}'.format(location,subj_extremes[f]['min']))
                subj_extremes[f]['min'] = location

            if location > subj_extremes[f]['max']:
                #print('{} is larger than {}'.format(location,subj_extremes[f]['max']))
                subj_extremes[f]['max'] = location

        for k in files[f]['files']:
            ds3 = pydicom.dcmread(k)
            data = ds3.pixel_array
            new_ds = _normalize(ds3, subj_extremes[f]['max'], k)
            new_ds.save_as(k)
        #print('normalized z location of {}'.format(f))
    print('finished renormalization of slice location')

def _rescale(img_raw):
    img_raw[img_raw == -2000] = 0  
    max_val = np.max(img_raw)
    img_new = ((img_raw / max_val)*2 -1).astype(np.float32)
    return img_new

def pixel_windowing(rescaled_pixel_value, window):
    max_density = window[1]
    min_density = window[0]
    if (rescaled_pixel_value == 1) or (rescaled_pixel_value>=max_density):
        new_value = max_density
    elif (rescaled_pixel_value == -1) or (rescaled_pixel_value <= min_density):
        new_value = min_density
    else:
        alpha = (rescaled_pixel_value - min_density)/(max_density-min_density)
        # new_value = rescaled_pixel_value*((max_density-min_density)/2)
        # new_value = rescaled_pixel_value - (rescaled_pixel_value - min_density)*min_density/max_density
        new_value = (alpha*max_density) + (1-alpha)*min_density
    return new_value


def slice_windowing(slice, window):
    new_data = np.empty((512, 512))
    for i in range(512):
        for j in range(512):
            new_data[i,j]=pixel_windowing(slice[i,j], window)
    return new_data

def _control_threshold(files):
    """
    Function to control appropriate threshold values
    """
    images = []
    subj_done = defaultdict(lambda:0)
    c = 0
    while c < 9:
        for f in files:
            subj = {}
            for i in files[f]['files']:
                if i.endswith('.dcm_0') and subj_done[f] != 1:
                    ds = pydicom.dcmread(i)
                    location = ds.SliceLocation
                    subj[i] = location
                    img_raw = ds.pixel_array
                    img_raw = _rescale(img_raw)
                    img_raw = slice_windowing(slice= img_raw, window=[-0.3, -0.2])
                    _,thresh = cv.threshold(img_raw,0.4,1,cv.THRESH_BINARY)
                    if thresh.any():
                    #if cv.countNonZero(thresh) > (img_raw.shape[0]*img_raw.shape[1])*0.07:
                        images.append([thresh, i[0:20]])
                        c+=1
                        subj_done[f]=1
    img_array = np.asarray(images)
    visualize(img_array)
    histo(img_array)

def reverse_order(files):
    """
    Swaps the ordering of the filenames according to their (normalized) heights

    args: normalized dcm file paths
    """
    print('start reversing order')
    # Initialize list with study batches to reverse
    reverse = []
    od = _order_slicelocation(files)
    for k in od:
        print('working on {}'.format(k))
        items = list(od[k].items())
        # obtain first and last slice of batch and perform some normalizations
        first = pydicom.dcmread(items[0][1])
        last = pydicom.dcmread(items[-1][1]) 
        p1 = first.pixel_array
        p2 = last.pixel_array
        if (p1.shape[0],p1.shape[1]) == (512,512) and (p2.shape[0],p2.shape[1]) == (512,512):
            p1 = _rescale(p1)
            p1 = slice_windowing(slice= p1, window=[-0.3, -0.2])
            p2 = _rescale(p2)
            p2 = slice_windowing(slice= p2, window=[-0.3, -0.2])
            # binarize images
            _,thresh1 = cv.threshold(p1,0.1,1,cv.THRESH_BINARY)
            _,thresh2 = cv.threshold(p2,0.1,1,cv.THRESH_BINARY)
            # check if more than 7% of thresholded images are non-black values
            # assuming at the top we have more black values (since only a small fraction
            # of the skull is shown, the picture with more black values is the top image
            cond1 = cv.countNonZero(thresh1) > (p1.shape[0]*p2.shape[1])*0.07
            cond2 = cv.countNonZero(thresh2) > (p1.shape[0]*p2.shape[1])*0.07
            if cond1 and cond2:
                if cv.countNonZero(thresh1) > cv.countNonZero(thresh2):
                    pass
                else:
                    reverse.append(k)
            elif cond2:
                reverse.append(k)
            else:
                pass
    print('finished reverting order')

    # revert order of all study batches that need reordering
    for f in files:
        # re-initialize subject dict
        subj = {}
        if f in reverse:
            print('reverse order for {}'.format(f))
            for i in files[f]['files']:
                ds = pydicom.dcmread(i)
                location = ds.SliceLocation
                subj[location] = i
            od2 = OrderedDict(sorted(subj.items()))
            items = list(od2.items())
            half = int(round(len(items)/2,0))
            for i in range(1,half+1):
                pair1 = pydicom.dcmread(items[i-1][1])
                loc1 = pair1.SliceLocation
                filename1 = items[i-1][1]
                pair2 = pydicom.dcmread(items[-i][1])
                loc2 = pair2.SliceLocation
                filename2 = items[-i][1]
                pair1.SliceLocation = loc2
                pair2.SliceLocation = loc1
                #print('swap {} with {}'.format(filename1,filename2))
                pair1.save_as(filename2)
                pair2.save_as(filename1)
    print('finished reverting order')


def _order_slicelocation(files):
    """
    args: dictionary of study batch with corresponding file names

    returns dict with study batch as key and orderd dict of all filenames
    according to their z-value as values
    """
    subj = {}
    all_od = {}
    for f in files:
        subj = {}
        for i in files[f]['files']:
            ds = pydicom.dcmread(i)
            location = ds.get('SliceLocation', "(missing)")
            subj[location] = i
        od = OrderedDict(sorted(subj.items()))
        all_od[f]=od

    return all_od

def rename(files):
    """
    Renames all files according to their z-location

    The new number in the file name is proportional to the z-value
    e.g. x000004.dcm has a lower z-location than x000010.dcm
    """
    print('start renaming')
    images = []
    subj_loc_ordered = _order_slicelocation(files)
    for k in subj_loc_ordered:
        items = list(subj_loc_ordered[k].items())
        # save each dcm in the batch with a new filename that 
        # corresponds to order of slice locations
        for dcm in items:
            index = items.index(dcm)
            ds = pydicom.dcmread(dcm[1])
            new_name = ("{}CT1{:05d}.dcm".format(dcm[1][:-12],index))
            ds.save_as(new_name)
    print('finished renaming')

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def create_batch(files):
    subj_loc_ordered = _order_slicelocation(files)
    for k in subj_loc_ordered:
        items = list(subj_loc_ordered[k].items())
        new_batch = chunks(items, int(len(items)/3)+1)
        batch_list = list(new_batch)
        for batch in batch_list:
            for _,file in batch:
                index = batch_list.index(batch)
                ds = pydicom.dcmread(file)
                new_name = file[:-9]+str(index+1)+file[-8:]
                ds.save_as(new_name)
    print('created batches')

def histo(image_array):
    """
    Get histograms of an array of images
    """
    w=512
    h=512
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns*rows +1):
        img = image_array[i-1][0]
        a = fig.add_subplot(rows, columns, i)
        plt.hist(img.ravel(),50,[0,1])
        #a.set_title("location "+str(image_array[i-1][1])[0:8])  # set title
        #plt.imshow(img)
    plt.tight_layout()
    plt.show()
            
def visualize(image_array):
    """
    Plot images of an array of images
    """
    w=512
    h=512
    fig=plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns*rows +1):
        img = image_array[i-1][0]
        a = fig.add_subplot(rows, columns, i)
        a.set_title(str(image_array[i-1][1])[-25:-12])  # set title
        plt.imshow(img)
    plt.tight_layout()
    plt.show()

def get_0_images(files):
    images = []
    subj = defaultdict(lambda:0)
    for f in files:
        if len(files[f]['files'])>5:
            for i in files[f]['files']:
                    ds = pydicom.dcmread(i)
                    if subj[f] != 1:
                        img_raw = ds.pixel_array
                        images.append([img_raw, i])
                        subj[f] = 1
    image_array = np.asarray(images)
    return image_array

def create_dicts(files):
    """
    Function to pickle transformed images.
    """

    all_images = {}
    for f in files:
        img_study = []
        for i in files[f]['files']:
            ds = pydicom.dcmread(i)
            img_raw = ds.pixel_array
            img_raw = _rescale(img_raw)
            img_study.append(img_raw)
        all_images[f] = img_study
    return all_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renormalize slice locations  \
        and reorder filenames according to slice heigth')

    parser.add_argument('-p', '--path',
                        help='Path where to start searching', default='.')

    args = parser.parse_args()
    #files = get_dcms(args.path)
    # initialize normalization
    #renormalize_z(files)
    #rename(files)
    files_new = get_dcms(args.path, re.compile(r'1\d{5}.dcm'))
    #print(files_new)
    reverse_order(files_new)
    # plot some results
    #images = get_0_images(files_new)
    #visualize(images)
    create_batch(files_new)
    #files_new = get_dcms(args.path, re.compile(r'11\d{4}.dcm'))
     #conrtol_threshold(files_new)
    files_btm = get_dcms(args.path, re.compile(r'11\d{4}.dcm'))
    visualize(get_0_images(files_btm))
    dict_0 = create_dicts(files_btm)
    with open('btm_slices.p', 'wb') as fp:
        pickle.dump(dict_0, fp, protocol=pickle.HIGHEST_PROTOCOL)
    files_mid = get_dcms(args.path, re.compile(r'12\d{4}.dcm'))
    dict_1 = create_dicts(files_mid)
    with open('btm_slices.p', 'wb') as fp:
        pickle.dump(dict_1, fp, protocol=pickle.HIGHEST_PROTOCOL)
    files_top = get_dcms(args.path, re.compile(r'13\d{4}.dcm'))
    dict_2 = create_dicts(files_top)
    with open('btm_slices.p', 'wb') as fp:
        pickle.dump(dict_2, fp, protocol=pickle.HIGHEST_PROTOCOL)