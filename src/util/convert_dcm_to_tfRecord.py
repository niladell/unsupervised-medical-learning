from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import re
import argparse

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _input_files(path):
    """Extract all the dcm file names in the dataset folder.

    Args:
        path (str): root path where to start searching
        filename_pattern (str): Any file with this substring will be
            selected

    Returns:
        (list): List of all files found (with the path from root)
    """
    files = []
    for p, d, folder in tf.gfile.Walk(path):
        #print(' Folder walk {}, {}, {}'.format(p, d, folder))
        for f in folder:
            if '.dcm' in f:
                files.append(f)
    return files

def load_all_data(path, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    files = _input_files(path)

    for i in files:    
        ds = pydicom.dcmread(i)
        height = ds.Rows
        width = ds.Columns
        #print(ds)
        if ds.PixelData:
            img = ds.PixelData


        location = ds.get('SliceLocation', "(missing)")
        identity = ds.get('PatientID')
        identity_nr = re.findall(r'\d+$', identity)


        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img),
            'identity_nr' : _int64_feature(int(identity_nr[0])),
            'slice_location' : _float_feature(location)}))
            #}))

        writer.write(example.SerializeToString())
    writer.close()

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

    load_all_data(args.path, args.output)