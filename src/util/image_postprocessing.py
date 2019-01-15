from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import psutil

def save_array_as_image(array, filename, do_histogram=True, bins=100):
    """Converts a numpy array to a PIL Image and undoes any rescaling.
    For more documentation on the several modes: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes

    Arrays are expected to have format "HWC"
    """
    try:
        if array.shape[2]==3:
            img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
        # img = Image.fromarray(array, mode='RGB')
        elif array.shape[2]==1:
            # PIL also provides limited support for a few special modes, including LA (L with alpha)
            # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
            img = Image.fromarray(np.uint8((np.squeeze(array) + 1.0) / 2.0 * 255), mode='L')
        else:
            raise ValueError("Don't know how to handle: {}".format(array.shape))

        with tf.gfile.Open(filename, 'w') as f:
            img.save(f, format='png')

    except MemoryError as e:
        tf.logging.error('Save image returned %s. Switching to alternative mode.', e)
        tf.logging.error('Memory usage at {}'.format(psutil.virtual_memory()))

        with tf.gfile.Open(filename, 'w') as f:
            np.save(f, array)
        #     plt.imsave(f, format='png')

        # array = array[:,:,0]
        # plt.figure()
        # plt.title('PLT: {}'.format(filename))
        # plt.imshow(array, cmap='gray')

    if do_histogram:
        # Plot of the rescaled image, before applying windowing:
        # plt.figure()  # It's kinda neat to have all the hists reported
        plt.hist(array.flatten(), bins=bins)
        plt.title('HIST: %s' % filename)
        filename = filename.split('.')
        filename = '.'.join(filename[:-1] + ['_hist'] + [filename[-1]])
        with tf.gfile.Open(filename, 'w') as f:
            plt.savefig(f, format='png')
        # plt.close() # It's kinda neat to have all the hists reported


    return img


def slice_windowing(img, window, up_val=1, low_val=0):
    """Apply a determined window to an image

    Args:
        img (numpy.array): Image
        window: (low bound of window, up bound of window)
        up_val (int, optional): Defaults to 1. Maximum output value
        low_val (int, optional): Defaults to 0. Minimum output value

    Returns:
        Windowed image.
    """
    min_density = window[0]
    max_density = window[1]

    alpha = (img - min_density) / (max_density - min_density)
    alpha[alpha<0]=0
    alpha[alpha>1]=1

    img = alpha * up_val + (1-alpha) * low_val
    # img = alpha * max_density + (1-alpha) * min_density

    return img

def save_windowed_image(array, filename):
    windowed_rescaled_slice = slice_windowing(img=array, window=[-0.3, -0.2], low_val=-1, up_val=+1)
    save_array_as_image(windowed_rescaled_slice, filename, do_histogram=False)
