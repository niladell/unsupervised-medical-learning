"""Convert CIFAR-like (plan to be expanded) datasets into TFRecod files."""

import tensorflow as tf
import numpy as np
import pickle
import argparse


def _unpickle(filename):
    """Unpicke batch CIFAR10 files

    Args:
        filename (str): File name

    Returns:
        (dict): Extracted data from files
    """

    with open(filename, 'rb') as fo:
        raw = pickle.load(fo, encoding='bytes')
    return raw


def _input_files(path, filename_pattern):
    """Extract all the CIFAR-10 file names in the dataset folder.

    Args:
        path (str): root path where to start searching
        filename_pattern (str): Any file with this substring will be
            selected

    Returns:
        (list): List of all files found (with the path from root)
    """
    print(path)
    files = []
    for p, d, folder in tf.gfile.Walk(path):
        print(' Folder walk {}, {}, {}'.format(p, d, folder))
        for f in folder:
            if filename_pattern in f:
                files.append(path + f)
    return files


#  We store our data either as int64 (single int), float (single float), or
# bytes. For anything different than single numbers (arrays, strings...),
# we'll encode the data as bytes and then store it.
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def load_all_data(path, filename_pattern, output_file):
    files = _input_files(path, filename_pattern)
    assert bool(files)  # check if files is not an empty list
    images = None
    labels = []
    for i in files:
        # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        raw = _unpickle(i)
        labels += raw[b'labels']
        img = raw[b'data']
        images = np.concatenate((images, img), axis=0) \
            if images is not None else img

    channels, height, width = (3, 32, 32)  # images.shape
    samples = images.shape[0]
    assert channels*height*width == images.shape[1]

    images = images.flatten().tobytes()
    labels = bytes(labels)

    ds = tf.train.Example(features=tf.train.Features(feature={
        'images':   _bytes_feature(images),
        'labels':   _bytes_feature(labels),
        'channels': _int64_feature(channels),
        'height':   _int64_feature(height),
        'width':    _int64_feature(width),
        'samples':  _int64_feature(samples)
    }))

    with tf.python_io.TFRecordWriter(output_file) as writer:
        writer.write(ds.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform original CIFAR \
        dataset into TFRecord format')
    parser.add_argument('-o', '--output',
                        help='Output TFRecord file', required=True)
    parser.add_argument('-p', '--path',
                        help='Path where to start searching', default='.')
    parser.add_argument('-i', '--filename',
                        help='File pattern to look for', required=True)
    args = parser.parse_args()

    load_all_data(args.path, args.filename, args.output)
