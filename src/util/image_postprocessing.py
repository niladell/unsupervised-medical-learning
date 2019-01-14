from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def save_array_as_image(array, filename):
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

        array = array[:,:,0]
        plt.figure()
        plt.title('PLT: {}'.format(filename))
        plt.imshow(array, cmap='gray')
        with tf.gfile.Open(filename, 'w') as f:
            plt.imsave(f, format='png')

    return img
