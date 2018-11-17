
import tensorflow as tf
from tensorflow.data import Dataset

class DataManager(object):
    """
    We create a DataManager.
    (still unclear what's the best way to implement this as a very eficient modula element)
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
