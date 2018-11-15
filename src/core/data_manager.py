
import tensorflow as tf
from tensorflow.data import Dataset

class DataManager(Dataset):
    """
    We create a DataManager based on the tf.data.Dataset class.
    """

    def preprocess_on_load_(self):
        pass

    def preprocess_on_batch_(self):
        pass

    def augment_data_strategy(self):
        pass

    def load_data(self):
        """
        Load data function
        """
