from absl import logging

import tensorflow as tf

from .core_datamanager import DataManager

class CoreModel(object):

    def __init__(self,
                 tf_session: tf.Session,
                 data_manager: DataManager,
                 learning_rate: float = 0.1
                  ):
        self.session = tf_session
        self.data_manager = data_manager

        self.learning_rate = learning_rate

    def define_model(self, data_source: DataManager , mode: str):
        """
        Definition of the model to use. Do not modify the function here
        placeholder for the actual definition in model/ (see example)

        Args:
            data_source (DataManager): Data manager object for the input data
            mode (str): Training and testing? # TODO Properly implement

        Raises:
            NotImplementedError: Model has to be implemented yet (in a separate instance in model/)
        """

        raise NotImplementedError('No model defined.')

    def build_model(self, mode: str):
        outputs, losses, others = self.define_model(self.data_manager.datasource, mode=mode)

        self.outputs = outputs
        self.losses = losses
        self.others = others

        # TODO Add routine to save

        self._build_optimizer()

    def _build_optimizer(self, optimizer_to_use=tf.train.AdamOptimizer):
        self.optimize_ops = []
        for loss in self.losses:  # TODO Create apropoiate external training scheme
            optimize_op = optimizer_to_use(
                learning_rate=self.learning_rate
            ).minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
            self.optimize_ops.append(optimize_op)
        logging.info('Optimizers built')

    def train(self, steps):
        # Initialize or check if checkpoint # TODO add checkpoint manager
        self.session.run(tf.global_variables_initializer())

        for step in range(steps):  # TODO start from checkpoint steps
            # TODO TRAIN
            self.session.run(self.optimize_ops)
            if step % 100 == 0:
                logging.info('Step {} -- Accuracy {}'.format(step, self._validate()))

        logging.info('Done training.')

    def _validate(self):
        accuracy = self.session.run(self.others) # TODO Hacky workaround for testing
        return accuracy

    def evaluate(self):
        pass
