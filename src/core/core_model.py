import tensorflow as tf

from .data_manager import DataManager

class CoreModel(object):

    def __init__(self,
                 tf_session,
                  ):
        pass

    def define_model(self):
        """
        Definition of the model to use. Do not modify the function here
        placeholder for the actual definition in model/ (see example)

        Raises:
            NotImplementedError: Model has to be implemented yet (in a separate instance in model/)
        """
        raise NotImplementedError('No model defined.')

    def build_model(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
