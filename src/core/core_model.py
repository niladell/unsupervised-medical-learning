from absl import logging

import tensorflow as tf

from .core_datamanager import DataManager

class CoreModel(object):

    def __init__(self,
                 tf_session: tf.Session,
                 training_dataset: DataManager = None,
                 validation_dataset: DataManager = None,
                 learning_rate: float = 0.1
                  ):
        self.session = tf_session
        self.dataset = {}
        if training_dataset: self.dataset['train'] = training_dataset.datasource
        if validation_dataset: self.dataset['validation'] = validation_dataset.datasource

        self._train_model =  True if training_dataset is not None else False
        self._validate_model = True if validation_dataset is not None else False
        self.learning_rate = learning_rate

    def define_model(self, data_source: DataManager , mode: str): #pylint: disable=E0202
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

    def build_model(self):
        """ Build the model. """

        # This overwrites define_model, is that ok?
        self.define_model = tf.make_template(self.define_model.__name__, self.define_model, create_scope_now_=True)  #pylint: disable=E1101

        self.outputs = {}
        self.losses = {}
        self.otters = {}

        def _build(mode):
            outputs, losses, others = self.define_model(data_source=self.dataset[mode], mode=mode)
            self.outputs[mode] = outputs
            self.losses[mode] = losses
            self.otters[mode] = others
            self._build_optimizer()

        if self._train_model:
            _build('train')

        if self._validate_model:
            _build('validation')

        # TODO Add routine to save

    def _build_optimizer(self, optimizer_to_use=tf.train.AdamOptimizer):
        self.optimize_ops = []
        for loss in self.losses['train']:  # TODO Create apropoiate external training scheme
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

        fetches = {}
        fetches['optimize_ops'] = self.optimize_ops
        fetches['losses'] = self.losses['train']
        if self.otters['train']:
            fetches['others'] = self.otters['train']

        for step in range(steps):  # TODO start from checkpoint steps
            # TODO TRAIN
            train_out = self.session.run(fetches=fetches) # Training output not used for now #TODO Add to summary
            if step % 100 == 0:
                logging.info('Step {} -- Validation result: {}'.format(step, self._validate()))

        logging.info('Done training.')

    def _validate(self):
        fetches = {}
        fetches['losses'] = self.losses['validation']
        if self.otters['train']:
            fetches['others'] = self.otters['validation']

        validation_out = self.session.run(fetches=fetches)
        return validation_out

    def evaluate(self):
        pass
