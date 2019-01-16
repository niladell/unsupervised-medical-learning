"""Prototype run file

Example run command:
python src/main.py --model_dir=gs://[BUCKET_NAME]/cifar10/outputs --data_dir=gs://[BUCKET_NAME]/cifar10/data  --tpu=[TPU_NAME] --dataset=[DATASET]
"""

import sys
import logging
logging.basicConfig(format='%(filename)s: '
                           '%(levelname)s: '
                           '%(funcName)s(): '
                           '%(lineno)d:\t'
                           '%(message)s')
from absl import flags
import tensorflow as tf
# TODO ¿move all to normal logging module?
# tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS


# Cloud TPU Cluster Resolvers
flags.DEFINE_boolean('use_tpu', True, 'Use TPU for training')
flags.DEFINE_string(
    'tpu', default='node-1',
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default='us-central1-f',
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_integer('num_shards', None, 'Number of TPU chips')

# Model specific paramenters
flags.DEFINE_string('model_dir', '', 'Output model directory')
flags.DEFINE_string('data_dir', '', 'Dataset directory')
flags.DEFINE_string('model', 'BASIC',
                    "Which model to use. Avail: 'BASIC' 'RESNET'")
flags.DEFINE_string('dataset', 'CIFAR10',
                    "Which dataset to use. Avail: 'CIFAR' 'CELEBA'")
flags.DEFINE_integer('noise_dim', 64,
                     'Number of dimensions for the noise vector')
flags.DEFINE_string('g_optimizer', 'ADAM', 'Optimizer to use for the'
                     'generator (now supported: ADAM')
flags.DEFINE_string('d_optimizer', 'SGD', 'Optimizer to use for the'
                     'discriminator (now supported: SGD, ADAM')

flags.DEFINE_boolean('use_encoder', False, 'Use an encoder')
flags.DEFINE_string('encoder', 'ATTACHED', 'Type of encoder to use.'
                    'Options are "ATTACHED" or "INDEPENDENT"')
flags.DEFINE_string('e_optimizer', 'ADAM',  'Optimizer to use for the'
                     'encoder (now supported: SGD, ADAM')

flags.DEFINE_integer('batch_size', 1024,
                     'Batch size for both generator and discriminator')
flags.DEFINE_float('soft_label', 0.2,
                    'Perturb (i.e. soften) labels randomly between -+ this' +\
                    'value. Soft label of 0 is the same as hard labels {0,1}')
flags.DEFINE_float('wgan_lambda', 10.0,
                    'Lambda value of WGAN penalty. See Improved WGAN ' +\
                    '(arXiv:1704.00028). Default value 10 (from paper)')
flags.DEFINE_boolean('use_wgan', True,
                    'Whether to use WGAN penalty or not')
flags.DEFINE_integer('wgan_n', 5,
                     'In WGAN penalty: Number of times that the Discriminator ' +\
                     ' (critic) is updated per step')

flags.DEFINE_float('e_loss_lambda', 1.0,
                    'Factor by which the encoder loss is scaled (`Loss = ' +\
                    'Adv_loss + lambda * Enc_loss`)')
flags.DEFINE_integer('train_steps', 50000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 1000,
                     'Steps per eval and image generation')
flags.DEFINE_integer('num_eval_images', 1024,
                     'Number of images on the evaluation')
flags.DEFINE_integer('num_viz_images', 100,
                     'Number of images generated on each PREDICT')
flags.DEFINE_integer('iterations_per_loop', 200,
                     'Steps per interior TPU loop. Should be less than'
                     ' --train_steps_per_eval')
flags.DEFINE_float('learning_rate', 0.0002, 'LR for both D and G')
flags.DEFINE_boolean('eval_loss', True,
                     'Evaluate discriminator and generator loss during eval')

flags.DEFINE_string('log_level', 'INFO', 'Logging level')
flags.DEFINE_boolean('ignore_params_check', False,
                    'Ignores checking parameters and overwrites params.txt')

if __name__ == "__main__":
    FLAGS(sys.argv)
    log = logging.getLogger('tensorflow')
    log.setLevel(FLAGS.log_level)

    # Get the model and dataset # TODO there has to be a better way right?
    if FLAGS.model.upper() == 'BASIC':
        from model import BasicModel as Model
    elif FLAGS.model.upper() == 'RESNET':
        from model import ResModel as Model
    else:
        raise NameError('{} is not a proper model name.'.format(FLAGS.model))
    if FLAGS.dataset.upper() == 'CIFAR10':
        from datamanager.CIFAR_input_functions import generate_input_fn
    elif FLAGS.dataset.upper() == 'CELEBA':
        from datamanager.celebA_input_functions import generate_input_fn
    elif FLAGS.dataset.upper() == 'CQ500':
        from datamanager.cq500_input_functions import generate_input_fn
    elif FLAGS.dataset.upper() == 'CQ500_256':
        from datamanager.cq500_256_input_functions import generate_input_fn

    else:
        raise NameError('{} is not a proper dataset name.'.format(FLAGS.dataset))


    ##### START
    model = Model(model_dir=FLAGS.model_dir, data_dir=FLAGS.data_dir, dataset=FLAGS.dataset,
                # Model parameters
                learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, noise_dim=FLAGS.noise_dim,
                soft_label_strength=FLAGS.soft_label,
                # WGAN
                use_wgan_penalty=FLAGS.use_wgan, wgan_lambda=FLAGS.wgan_lambda, wgan_n=FLAGS.wgan_n,
                # Encoder
                use_encoder=FLAGS.use_encoder, encoder=FLAGS.encoder, e_loss_lambda=FLAGS.e_loss_lambda,
                # Optimizers
                g_optimizer=FLAGS.g_optimizer, d_optimizer=FLAGS.d_optimizer, e_optimizer=FLAGS.e_optimizer,
                # Training and prediction settings
                iterations_per_loop=FLAGS.iterations_per_loop, num_viz_images=FLAGS.num_viz_images,
                # Evaluation settings
                eval_loss=FLAGS.eval_loss, train_steps_per_eval=FLAGS.train_steps_per_eval,
                num_eval_images=FLAGS.num_eval_images,
                # TPU settings
                use_tpu=FLAGS.use_tpu, tpu=FLAGS.tpu, tpu_zone=FLAGS.tpu_zone,
                gcp_project=FLAGS.gcp_project, num_shards=FLAGS.num_shards,
                ignore_params_check=FLAGS.ignore_params_check)

    model.save_samples_from_data(generate_input_fn)
    model.build_model()
    model.train(FLAGS.train_steps, generate_input_fn)
    tf.logging.info('Finished training.')



    # Bit of script to run the encoding on a test image
    # TODO create proper file to analyze images
    encode_test_img = False
    if encode_test_img:
        from scipy.misc import imread
        from util.image_preprocessing import center_crop

        im = imread('others/img.jpg')
        im = center_crop(im, 64, 64)

        encoded_img = model.encode(im)
        print('\n\n Encoded img: \n{}'.format(encoded_img))