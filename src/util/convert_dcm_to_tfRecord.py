from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pydicom


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_outfile = 'test.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_outfile)

filename = 'ct_test.dcm'
ds = pydicom.dcmread(filename)
img_raw = ds.PixelData

height = ds.pixel_array.shape[0]
width = ds.pixel_array.shape[1]
    
example = tf.train.Example(features=tf.train.Features(feature={
    'height': _int64_feature(height),
    'width': _int64_feature(width),
    'image_raw': _bytes_feature(img_raw)}))

print(example)    
writer.write(example.SerializeToString())

writer.close()