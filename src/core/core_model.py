from absl import logging

import tensorflow as tf

from .core_datamanager import DataManager

class CoreModel(object):

    def __init__(self,
                 tf_session: tf.Session,
                 learning_rate: float,
                 training_dataset: DataManager = None,
                 validation_dataset: DataManager = None,
                 output_path: str = './outputs'
                ):
        self.session = tf_session
        self.output_path = output_path
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
        self.define_model = tf.make_template(self.define_model.__name__,  #pylint: disable=E1101
                                             self.define_model,
                                             create_scope_now_=True)

        self.outputs = {}
        self.losses = {}
        self.otters = {}

        def _build(mode):
            outputs, losses, others = self.define_model(data_source=self.dataset[mode], mode=mode)
            self.outputs[mode] = outputs
            self.losses[mode] = losses
            self.otters[mode] = others
            if mode == 'train':
                self._build_optimizer()

        # TODO Move clean and summary to proper section
        self.summary_ops = {}
        if self._train_model:
            _build('train')
            summary = []
            for idx, loss in enumerate(self.losses['train']):
                summary.append(
                    tf.summary.scalar(name='train/loss_{}'.format(idx), tensor=loss))
            for idx, element in enumerate(self.otters['train']):
                summary.append(
                    tf.summary.scalar(name='train/otter_{}'.format(idx), tensor=element))
            self.summary_ops['train'] = tf.summary.merge(summary)

        if self._validate_model:
            _build('validation')
            summary = []
            for idx, loss in enumerate(self.losses['validation']):
                summary.append(
                    tf.summary.scalar(name='val/loss_{}'.format(idx), tensor=loss))
            for idx, element in enumerate(self.otters['validation']):
                summary.append(
                    tf.summary.scalar(name='val/otter_{}'.format(idx), tensor=element))
            self.summary_ops['validation'] = tf.summary.merge(summary)

        self.writer = tf.summary.FileWriter(self.output_path,
                                            self.session.graph)

        # TODO Add routine to save
        logging.info('Model construction complete.')

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
        # fetches['losses'] = self.losses['train']
        # if self.otters['train']:
        #     fetches['others'] = self.otters['train']
        fetches['summary_ops'] = self.summary_ops['train']

        for step in range(steps):  # TODO start from checkpoint steps
            # TODO clean code and optimize ops
            train_out = self.session.run(fetches=fetches)
            self.writer.add_summary(train_out['summary_ops'], global_step=step)
            if step % 50 == 0: # TODO every how many steps? Automate?
                logging.info('Step {} -- Validation result: {}'.format(step, self._validate(step)))

        logging.info('Done training.')

    def _validate(self, global_step):
        """ Run network on validation set """
        # Todo clean summaries and add example outputs
        fetches = {}
        fetches['losses'] = self.losses['validation']
        if self.otters['train']:
            fetches['others'] = self.otters['validation']
        fetches['summary_ops'] = self.summary_ops['validation']
        validation_out = self.session.run(fetches=fetches)
        self.writer.add_summary(validation_out['summary_ops'], global_step=global_step)
        del validation_out['summary_ops']
        return validation_out

    def evaluate(self):
        pass
