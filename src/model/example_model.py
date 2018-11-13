import tensorflow as tf

from core import CoreModel

class ExampleModel(CoreModel):
    """
    Example definition of a model/network architecture using this template.
    """


    def define_net(self, inputs):
        """
        Example definition of a network
        
        Args:
            inputs ([list of tf.Tensor]): Data passed to the model.
        
        Returns:
            [tuple]: (1) Function loss to optimize, (2) Prediction made by the netowrk
        """


        prediction = inputs
        loss = prediction
        return loss, prediction