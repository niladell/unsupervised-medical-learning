from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
import re


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

tfrecords_outfile = 'test.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_outfile)

filename = 'ct_test.dcm'
ds = pydicom.dcmread(filename)
img_raw = ds.PixelData
print(ds)

height = ds.pixel_array.shape[0]
width = ds.pixel_array.shape[1]
location = ds.get('SliceLocation', "(missing)")
identity = ds.get('PatientID')
identity_nr = re.findall(r'\d+$', identity)
    
example = tf.train.Example(features=tf.train.Features(feature={
    'height': _int64_feature(height),
    'width': _int64_feature(width),
    'image_raw': _bytes_feature(img_raw),
    'identity_nr' : _int64_feature(int(identity_nr[0])),
    'slice_location' : _float_feature(location)}))


#print(example)    
writer.write(example.SerializeToString())

writer.close()


print([ds.pixel_array.min(),ds.pixel_array.max()])

#print([ds.pixel_array.min(),ds.pixel_array.max()])



#plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
#plt.show()
