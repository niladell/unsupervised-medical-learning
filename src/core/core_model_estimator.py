""" Core tensorflow model that basically encapsulates all the basic ops
    in order to run an experiment.
"""

import os
from absl import logging

import tensorflow as tf
from tensorflow.contrib import tpu

from .core_datamanager_estimator import DataManagerTPU as DataManager

class CoreModelTPU(object):

    def __init__(self,
                 tf_session: tf.Session,
                 learning_rate: float,
                 training_dataset: DataManager = None,
                 validation_dataset: DataManager = None,
                 output_path: str = '../outputs',
                 use_tpu: str = False,
                 tpu_name: list = [],
                 data_dir= '/mnt/iowa_bucket/cifar10/data/'
                ):
        self.data_dir = data_dir
        if output_path[-1] == '/':
            output_path = output_path[:-1]
        self.output_path = output_path + '/' + self.__class__.__name__

        self.session = tf_session

        # TODO Get rid of the .datasource thing
        self.dataset = {}
        # if training_dataset: self.dataset['train'] = training_dataset.datasource
        # if validation_dataset: self.dataset['validation'] = validation_dataset.datasource

        self.datasource = {}
        self.datasource['train'] = training_dataset
        self.datasource['validation'] = validation_dataset

        self._train_model =  True if training_dataset is not None else False
        self._validate_model = True if validation_dataset is not None else False
        self.learning_rate = learning_rate

        self.use_tpu = use_tpu


    def define_model(self, data_source: DataManager , mode: str): #pylint: disable=E0202
        """Definition of the model to use. Do not modify the function here
        placeholder for the actual definition in `model/` (see example)

        Args:
            data_source (DataManager): Data manager object for the input data
            mode (str): Training and testing? # TODO Properly implement

        Raises:
            NotImplementedError: Model has to be implemented yet (in a separate instance in model/)
        """

        raise NotImplementedError('No model defined.')

    def build_model(self):
        """ Build the model. """
        if self.use_tpu:
            self._tpu_build()
        else:
            self._regular_build()

    def _tpu_build(self):
        """Build with TPUEstimators for TPU usage"""
        def _define_model(features, labels, mode, params):
            data_source = (features, labels)
            self.outputs = {}
            self.losses = {}
            self.otters = {}
            outputs, losses, others = self.define_model(data_source, mode)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tpu.TPUEstimatorSpec(
                    mode=mode, loss=losses, eval_metrics=others)
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tpu.TPUEstimatorSpec(
                    mode=mode, predictions=outputs
                )
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.losses['train'] = losses
                self._build_optimizer(tpu_support=True)
                if not len(self.optimize_ops) == 1:
                    logging.error('Implementati Error: More than one optimizer defined')
                    logging.warning(' [*] Selecting only the first optimizer')
                return tpu.TPUEstimatorSpec(
                    mode=mode, loss=losses[0], train_op=self.optimize_ops[0]
                )

        tpu_name = ['node-1'] # TODO Bring outside
        tpu_iterations = 500  # TODO Bring outside
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name)

        run_config = tf.contrib.tpu.RunConfig(
            model_dir=self.output_path,
            cluster=tpu_cluster_resolver,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True),
            tpu_config=tpu.TPUConfig(tpu_iterations),
            )

        self.estimator = tpu.TPUEstimator(
            model_fn=_define_model,
            use_tpu=True,
            train_batch_size=32*4, #self.dataset['train'].batch_size,
            eval_batch_size=32*4, #self.dataset['validation'].batch_size,
            config=run_config,
            params={"data_dir": self.data_dir}
        )


    def _regular_build(self):
        """Normal build for CPU/GPU usage"""
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
        self.saver = tf.train.Saver()
        # TODO Add routine to save
        logging.info('Model construction complete.')

    def _build_optimizer(self, optimizer_to_use=tf.train.AdamOptimizer, tpu_support=False):
        """Buids the optimizer(s) to minimize the loss(es) of the model.

        Args:
            optimizer_to_use (tf optimizer, optional): Defaults to tf.train.AdamOptimizer. Which
                optimizer to use.
            tpu_support (bool, optional): Defaults to False. If the optimizer has to support shard
                optimier, required for TPU usage.
        """
        self.optimize_ops = []
        for loss in self.losses['train']:  # TODO Create apropoiate external training scheme
            optimize_op = optimizer_to_use(
                learning_rate=self.learning_rate
            )
            if tpu_support:
                optimize_op = tpu.CrossShardOptimizer(optimize_op)
            optimize_op = optimize_op.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
            self.optimize_ops.append(optimize_op)
        logging.info('Optimizers built')

    def train(self, steps, input_fn=None):
        if self.use_tpu:
            self._tpu_train(steps, input_fn)
        else:
            self._regular_train(steps)

    def _tpu_train(self, steps, input_fn):

        # def _input_fn(params):
        #     featuers, labels = self.datasource['train'].input_fn(params['batch_size'])
        #     return featuers, labels

        self.estimator.train(
            input_fn=input_fn,
            max_steps=steps)
        logging.info('Es ist train?')
        self.estimator.evaluate(
            input_fn=self.dataset['validation'],
            steps=steps/50
        )
        print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

    def _regular_train(self, steps):
        # Initialize or check if checkpoint # TODO add checkpoint manager
        self.session.run(tf.global_variables_initializer())
        initial_step = self._restore()

        fetches = {}
        fetches['optimize_ops'] = self.optimize_ops
        # fetches['losses'] = self.losses['train']
        # if self.otters['train']:
        #     fetches['others'] = self.otters['train']
        fetches['summary_ops'] = self.summary_ops['train']

        for step in range(initial_step, steps):  # TODO start from checkpoint steps
            # TODO clean code and optimize ops
            train_out = self.session.run(fetches=fetches)
            self.writer.add_summary(train_out['summary_ops'], global_step=step)
            if step % 50 == 0: # TODO every how many steps? Automate?
                val = self._validate(step)
                logging.info('Step {} -- Validation result: {}'.format(step, val))
            if step % 1000 == 0:  # For now just another arbitrary number (how heavy is saving?)
                self._save(step)
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

    def _save(self, step):
        """Save the model weights.

        Args:
            step (int): Training step.
        """

        output_path = self.output_path + '/checkpoints/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        self.saver.save(self.session, save_path=output_path,global_step=step)

    def _restore(self):
        """Restore the trained variables from the last stored checkpoint

        Returns:
            int: The training step when this model was saved.
        """

        output_path = self.output_path + '/checkpoints/'
        checkpoint = tf.train.latest_checkpoint(output_path)
        if checkpoint:
            self.saver.restore(self.session, save_path=checkpoint)
            restored_step = int(checkpoint.split('-')[-1])  # Robust enough?
            return restored_step
        logging.info('Starting training from scratch.')
        return 0

    def evaluate(self):
        pass
