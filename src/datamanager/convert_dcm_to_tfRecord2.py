from PIL import Image
import numpy as np
# import skimage.io as io
import tensorflow as tf
import pydicom
import os


def get_list_of_dcm_path(txt_path):
    with open(txt_path, "r") as file:
        lines = file.read().split('\n')
        return lines

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convertToTfRecord(list_of_dcm_paths):
    for filepath in list_of_dcms:
        filepath = list_of_dcms[9]
        ds = pydicom.dcmread(filepath)
        img_raw = ds.PixelData

        height = ds.pixel_array.shape[0]
        width = ds.pixel_array.shape[1]

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw)}))

        tfrecords_outfile = filepath+'.tfrecords'

        writer = tf.python_io.TFRecordWriter(tfrecords_outfile)
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    dirname = '/Users/ines/Documents/Ensino superior/Masters in NSC ETH UZH/Deep Learning/Project/unsupervised-medical-learning/src/datamanager'
    list_of_dcms = get_list_of_dcm_path(dirname + '/list_of_dcms.txt')
    convertToTfRecord(list_of_dcms)
    # print(list_of_dcms)
