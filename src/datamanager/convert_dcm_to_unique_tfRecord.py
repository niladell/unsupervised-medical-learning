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

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convertToTfRecord(list_of_dcm_paths):
    tfrecords_outfile = 'huge_Q500.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecords_outfile)


    for idx, filepath in enumerate(list_of_dcms):
        ds = pydicom.dcmread(filepath)
        img_raw = ds.pixel_array

        # If we use a mask we can ignore quite a number of outputs, and avoud
        # propagating useless stuff
        mask = np.full_like(img_raw, True)
        mask[img_raw==-2000] = False # Mask of the actual values of the scan
        img_raw[img_raw==-2000] = 0 #?¿ Values outside the scanner can become 0¿
        max_val = np.max(img_raw)
        # TODO wtf image_raw type:
        #    image_raw is type=np.int16, which means a range of +-32767. So far
        #    I have seen that there's no value <0 but for -2000 (out of the scan)
        #    which is a starting point, however I'm not sure un the upper bound.
        #    The biggest value so far was 5067 (on Inês' set of dcms)... meaning
        #    wat do we do with that?

        height = ds.pixel_array.shape[0]
        width = ds.pixel_array.shape[1]
        patientID = ds.PatientID
        z = float(ds.SliceLocation)
        dcmname = os.path.basename(os.path.normpath(filepath))
        example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    'filename': _bytes_feature(dcmname.encode()),
                    'id': _bytes_feature(patientID.encode()),
                    'z-slice': _float_feature(z),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'max_val': _int64_feature(max_val),
                    'image_raw': _bytes_feature(img_raw.tobytes()), #!! Needs to be reconstructed with int16!!!
                    'mask': _bytes_feature(mask.tobytes())}
                )
            )

        print('{}/{} -- {}: {} (z{})'.format(idx, len(list_of_dcms), patientID, dcmname, z))


        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    # dirname = '/Users/ines/Documents/Ensino superior/Masters in NSC ETH UZH/Deep Learning/Project/unsupervised-medical-learning/src/datamanager'
    list_of_dcms = get_list_of_dcm_path('./list_of_dcms.txt')
    convertToTfRecord(list_of_dcms)
    # print(list_of_dcms)
