import numpy as np
from skimage.transform import rescale
import tensorflow as tf
import pydicom
import os
import matplotlib.pyplot as plt


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


def convertToTfRecord(list_of_dcm_paths, out_file, scaling_factor, do_plot=False):
    tfrecords_outfile = out_file  # 'huge_Q500.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecords_outfile)

    for idx, filepath in enumerate(list_of_dcms):
        ds = pydicom.dcmread(filepath)
        img_raw = ds.pixel_array

        # If we use a mask we can ignore quite a number of outputs, and avoud
        # propagating useless stuff
        mask = np.full_like(img_raw, True)
        mask[img_raw == -2000] = False  # Mask of the actual values of the scan
        img_raw[img_raw == -2000] = 0  # ?¿ Values outside the scanner can become 0¿

        max_val = np.max(img_raw)
        # TODO check if that properly works --> this is the type tf uses in the model
        # TODO see if using the max of each image gives good results --> different way
        # of normaizing if we compared to what we do with unint (dividde 255) on the other
        # datasets (CIFAR10 and celebA)
        img_raw = (img_raw / max_val).astype(np.float32)
        # TODO wtf image_raw type:
        #    image_raw is type=np.int16, which means a range of +-32767. So far
        #    I have seen that there's no value <0 but for -2000 (out of the scan)
        #    which is a starting point, however I'm not sure un the upper bound.
        #    The biggest value so far was 5067 (on Inês' set of dcms)... meaning
        #    wat do we do with that?

        # Optional: resize image as 512 x 512 may be too much to train

        if do_plot:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img_raw, cmap='gray')
            ax1.set_title('Original')
            # I dont like this structur but this
            # continues below...

        height = img_raw.shape[0]
        width = img_raw.shape[1]
        if height != 512 and width != 512:
            print(' !!! Img {} has shape {}  -- file: {}'.format(idx, img_raw.shape, filepath), flush=True)
            continue

        if scaling_factor:
            img_raw = rescale(img_raw,
                              scale=1 / scaling_factor,
                              anti_aliasing=True,
                              clip=True)
                              # preserve_range=False)
            if img_raw.shape[0] != height / scaling_factor or \
               img_raw.shape[1] != width / scaling_factor:

                height = img_raw.shape[0]
                width = img_raw.shape[1]
                print(' !!! Img {} has shape {}  -- file: {}'.format(idx, img_raw.shape, filepath), flush=True)
                continue
        if do_plot:
            # ... continuation from above
            print(height, width)
            print(np.min(img_raw), np.max(img_raw))
            ax2.imshow(img_raw, cmap='gray')
            ax2.set_title('Downsample by {}'.format(scaling_factor))
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

        patientID = ds.PatientID
        z = float(ds.SliceLocation)
        dcmname = os.path.basename(os.path.normpath(filepath))
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'filename': _bytes_feature(dcmname.encode()),
                    'id': _bytes_feature(patientID.encode()),
                    'z-slice': _float_feature(z),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'max_val': _int64_feature(max_val),
                    'image': _bytes_feature(img_raw.tobytes()),  # !! Needs to be reconstructed with float32!!!
                    'mask': _bytes_feature(mask.tobytes())}
            )
        )
        if idx % 10 == 0:
            print('{}/{} -- {}: {} {} (z{})'.format(idx, len(list_of_dcms), patientID, dcmname,img_raw.shape, z), flush=True)

        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert dcms to tfrecords.')
    # parser.add_argument('-f', '--folder', type=str,
    #                     help='Folder with dcm files to be converted (recursive on subfolders)')
    parser.add_argument('-o', '--output', default='train.tfrecords',
                        help='Output file')
    parser.add_argument('-n', '--downsample', type=int, default=None)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    list_of_dcms = get_list_of_dcm_path('./list_of_dcms.txt')
    convertToTfRecord(list_of_dcms, args.output, args.downsample, args.plot)
