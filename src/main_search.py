"""Does the same as main.py but over a set of selected hyperparameters.

See notes and FLAGS on main.py on how to run.
"""

import sys
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager

from main import FLAGS # It doesn't seem the best way to rely on the main function for the flag definitions
from model import BasicModel
from datamanager.celebA_input_functions import generate_input_fn


def grid_search(Model, generate_input_fn, parameter_range, FLAGS, parallel_tpu=2):
    """Perform grid search of selected parameters"""

    params = FLAGS.flag_values_dict()

    def run(Model, generate_input_fn, parameters, tpu, proc_dict):
        time_start = time.time()
        # TODO Make for any possible key
        params['learning_rate'] = parameters # Testing with learning rate only
        model = BasicModel(model_dir=params['model_dir'], data_dir=params['data_dir'], dataset=params['dataset'],
            # Model parameters
            learning_rate=params['learning_rate'], batch_size=params['batch_size'], noise_dim=params['noise_dim'],
            # Encoder
            use_encoder=params['use_encoder'], encoder=params['encoder'],
            # Optimizers
            g_optimizer=params['g_optimizer'], d_optimizer=params['d_optimizer'], e_optimizer=params['e_optimizer'],
            # Training and prediction settings
            iterations_per_loop=params['iterations_per_loop'], num_viz_images=params['num_viz_images'],
            # Evaluation settings
            eval_loss=params['eval_loss'], train_steps_per_eval=params['train_steps_per_eval'],
            num_eval_images=params['num_eval_images'],
            # TPU settings
            use_tpu=params['use_tpu'], tpu='node-{}'.format(tpu), tpu_zone=params['tpu_zone'],
            gcp_project=params['gcp_project'], num_shards=params['num_shards'])

        model.save_samples_from_data(generate_input_fn)
        model.build_model()
        model.train(params['train_steps'], generate_input_fn)
        duration = time.time() - time_start
        tf.logging.info('End time {0}:{1:.2f}'.format(int(duration // 60), duration % 60))
        # return_dict[params['tpu']]=True # Needed to make things work in the future (with proper multiprocessing)


    #   TODO RUN everything in "proper" multiprocessesing. I'm stil not sure on how to recover the TPU form a
    # job that has finished and inmediately start a new one
    #   This is a start for now. It runs the processes in parallel but it needs all of them to finish in order to
    # start again.
    manager = Manager()
    return_dict = manager.dict()
    for i in range(1, parallel_tpu + 1):
        return_dict[i]=True
    jobs = {}
    counter = 0
    while counter < len(parameter_range):
        for tpu, free in return_dict.items():
            if free:
                return_dict[tpu] = False
                p = Process(
                    target=run,
                    args=(BasicModel, generate_input_fn, parameter_range[counter], tpu, return_dict)
                )
                jobs[tpu] = p
                counter += 1
                jobs[tpu].start()
        for tpu, job in jobs.items():
            job.join()
            return_dict[tpu] = True




if __name__ == "__main__":
    FLAGS(sys.argv)

    learning_range = [0.01, 0.001, 0.0005, 0.0001]  # TEST Ranges

    grid_search(BasicModel, generate_input_fn, learning_range, FLAGS)

