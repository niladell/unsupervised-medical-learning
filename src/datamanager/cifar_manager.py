from core import DataManager

import tensorflow as tf
from tensorflow.data import Dataset
from os import walk
import pickle
import numpy as np

# Download dataset
# cd dataset/
# wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# tar xvf cifar-10-python.tar.gz

class CIFAR10(DataManager):
    """
    Simple data manager

     > Disclaimer:
     >  STILL NOT SURE IF THE PROCESS USED HERE IS CORRECT... I'M NOT CONFIDENT AT ALL
     > ABOUT HOW I SHOULD PROPERLY HANDLE DATA HERE. SO THE FOLLOWING CODE MAY CHANGE A LOT
     > IN FUTURE COMMITS

    Args:
        DataManager ([type]): [description]

    Returns:
        [type]: [description]
    """


    def __init__(self,
                 tf_session  : tf.Session,
                 dataset_path: str = '../datasets/cifar-10-batches-py/',    # Path to the dataset
                 file_name   : str = 'data_batch',                          # Subsrting in files names (here: 'data_batch' <-- 'data_batch_1', 'data_batch_2', ...)
                 defaul_batch_size: int = 32,
                 ):
        self.path = dataset_path
        self.file_name = file_name
        self.files = []

        self.num_channels = 3
        self.img_size = 32
        self.num_classes = 10

        self.batch_size = defaul_batch_size
        self._session = tf_session

        self.pre_load()
        self.initialize()

    def initialize(self):
        """
        Initialize iteratior.

        # TODO Should the session object be passed here or be an atribute of the class
        """

        # sess(self.iteratior.initializer)

        # Initialize the iterator
        self._session.run(self.iterator.initializer)

        self.datasource = self.iterator.get_next()

    def _input_files(self):
        """
        Extract all the CIFAR-10 file names in the dataset folder.
        """

        files = []
        for _,_,folder in walk(self.path):
            for f in folder:
                if self.file_name in f:
                    files.append(f)
        self.files = files

    def _convert_images(self, raw):
        """
        -- From TF Tutorials --
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self.num_channels, self.img_size, self.img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            raw = pickle.load(fo, encoding='bytes')
        return raw

    def input_parser(self):
        """
        Parse CIFAR input files

        Returns:
            [np.array, list]: Returns the images and the lables
        """

        self._input_files()
        images = None
        labels = []
        for i in self.files:
            raw = self._unpickle(self.path + i) # -> dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
            labels += raw[b'labels']
            img = self._convert_images(raw[b'data'])
            images = np.concatenate((images, img), axis=0) if images is not None else img
        return images, labels

    def pre_load(self):
        """
        First loading step of the class: parse the data and generate the dataset instances.
        """
        # TODO Feels not 100% correct way of approaching it for most of the cases

        data = self.input_parser() # (images, labels)
        dataset = Dataset.from_tensor_slices(data)
        # Automatically refill the data queue when empty
        dataset = dataset.repeat()
        # Create batches of data
        dataset = dataset.batch(self.batch_size)
        # Prefetch data for faster consumption
        dataset = dataset.prefetch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
