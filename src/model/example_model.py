import tensorflow as tf
import tensorflow_hub as hub

from core import CoreModel

class ExampleModel(CoreModel):
    """
    Example definition of a model/network architecture using this template.
    """

    def define_model(self, inputs, mode):
        """
        Example definition of a network

        Args:
            inputs ([list of tf.Tensor]): Data passed to the model.
            mode ([str]): ['train', 'eval', 'predict']

        Returns:
            [tuple]: (1) Function loss to optimize, (2) Prediction made by the netowrk
        """

        """EXAMPLE TAKEN FROM TF WEBSITE"""
        training = True if mode == 'train' else False

        # Input Layer
        input_layer, labels =  inputs #  input shape -->  [?, 28, 28, 1]

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.layers.flatten(pool2)
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=training)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        outputs = {'prediction': logits, 'probabilities': tf.nn.softmax(logits, name='softmax_layer')}

        return outputs, [loss], {}  # outputs, losses, metrics/otters
        # TODO Temporal unconsistent formatting on the return elements: use dict, tf.somthing, list...?