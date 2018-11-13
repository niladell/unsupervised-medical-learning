from absl import flags
from absl import logging

import tensorflow_hub as hub

from core import DataManager
from model import ExampleModel

# TODO This has to be externalized -> parse args
flags.DEFINE_bool('use_tpu', True, 'Use TPU model instead of CPU.')
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_string('data', None, 'Path to training and testing data.')  # Pass to data manager
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored.'))


FLAGS = flags.FLAGS

# TODO Write main calls here