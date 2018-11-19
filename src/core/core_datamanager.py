
import tensorflow as tf
from tensorflow.data import Dataset

class DataManager(object):
    """
    We create a DataManager.
    (still unclear what's the best way to implement this as a very eficient modula element)
    """
    def __init__(self, tf_session  : tf.Session, batch_size: int = 32):
        self._session = tf_session
        self.batch_size = batch_size
        self._create_tf_datastream()

    def _create_tf_datastream(self):
        data = self.load_data()
        dataset = Dataset.from_tensor_slices(data)
        # Automatically refill the data queue when empty
        dataset = dataset.repeat()
        # Create batches of data
        dataset = dataset.batch(self.batch_size)
        # Prefetch data for faster consumption
        dataset = dataset.prefetch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        self._session.run(self.iterator.initializer)

        self.datasource = self.iterator.get_next()

    def load_data(self):
        """
        Load data function
        """
        raise NotImplementedError('Loading defined.')


    def preprocess_on_load_(self):
        pass

    def preprocess_on_batch_(self):
        pass

    def augment_data_strategy(self):
        pass

