
# coding: utf-8

# In[1]:


import sys
sys.path.append("src")


import random

import logging
logging.basicConfig(format='%(filename)s: '
                           '%(levelname)s: '
                           '%(funcName)s(): '
                           '%(lineno)d:\t'
                           '%(message)s')
from absl import flags
import tensorflow as tf

log = logging.getLogger('tensorflow')
log.setLevel('INFO')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


# In[2]:


def make_model(model_dir, reconstruction_loss):
    log.setLevel('INFO')
#     model_dir = 'outputs'
    data_dir = ''
    dataset = 'TOY'
    noise_dim= 2

    # GENERIC PARAMS
    batch_size= 1024
    d_optimizer= 'ADAM'

#     reconstruction_loss= True
    use_encoder= True

    e_loss_lambda= 0.0
    e_optimizer= 'ADAM'
    encoder= 'ATTACHED'
    eval_loss= True
    g_optimizer= 'ADAM'
    soft_label = 0

    window_lambda = 0
    use_wgan = 0
    gcp_project= None
    iterations_per_loop= 500
    lambda_window= 1.0
    learning_rate= 0.0002
    noise_cov= 'IDENTITY'
    num_eval_images= 1024
    num_shards= None
    num_viz_images= 100
    soft_label_strength= 0.2
    tpu= ''
    tpu_zone= 'us-central1-f'
    train_steps_per_eval= 1000
    use_tpu= False
    use_window_loss= False
    wgan_lambda= 10.0
    wgan_n= 5
    wgan_penalty= False

    ignore_params_check = False

    from model import ToyModel as Model
    from datamanager.toydist_input_functions import generate_input_fn


    try:
        import shutil
        shutil.rmtree('outputs')
    except FileNotFoundError:
        print('Folder outs does not exist')

    ##### START
    model = Model(model_dir=model_dir, data_dir=data_dir, dataset=dataset,
                # Model parameters
                learning_rate=learning_rate, batch_size=batch_size, noise_dim=noise_dim,
                noise_cov=noise_cov, soft_label_strength=soft_label,
                use_window_loss=use_window_loss, lambda_window=window_lambda,
                # WGAN
                use_wgan_penalty=use_wgan, wgan_lambda=wgan_lambda, wgan_n=wgan_n,
                # Encoder
                use_encoder=use_encoder, encoder=encoder, e_loss_lambda=e_loss_lambda,
                # ¯\_(ツ)_/¯
                reconstruction_loss=reconstruction_loss,
                # Optimizers
                g_optimizer=g_optimizer, d_optimizer=d_optimizer, e_optimizer=e_optimizer,
                # Training and prediction settings
                iterations_per_loop=iterations_per_loop, num_viz_images=num_viz_images,
                # Evaluation settings
                eval_loss=eval_loss, train_steps_per_eval=train_steps_per_eval,
                num_eval_images=num_eval_images,
                # TPU settings
                use_tpu=use_tpu, tpu=tpu, tpu_zone=tpu_zone,
                gcp_project=gcp_project, num_shards=num_shards,
                ignore_params_check=ignore_params_check)
    model.build_model()
    model.train(50000, generate_input_fn)
    log.setLevel('CRITICAL')


    import tensorflow as tf

    def get_sample(batch_size):
        def _input_fn(params):
             return {'random_noise':
                tf.constant([[np.random.uniform(0,1),np.random.uniform(0,1)] 
                                 for i in range(batch_size)], dtype=tf.float32)}

        return next(model.est.predict(_input_fn))['generated_images']

    samples = [get_sample(64) for i in tqdm(np.linspace(-1, 1,100))]
    samples = np.vstack(samples)

    NUMBER_OF_GAUSSIANS=10
    def sample_toy_distr():
        x = np.random.normal(0, 0.1)
        y = np.random.normal(0, 0.1)
        centers = [(i*5,j*10) for i in range(NUMBER_OF_GAUSSIANS) for j in range(NUMBER_OF_GAUSSIANS)]
        mu_x, mu_y = random.sample(centers,1)[0]
        return [x + mu_x, y + mu_y]

    r_samples = [sample_toy_distr() for i in tqdm(np.linspace(-1, 1, 10000))]
    r_samples = np.vstack(r_samples)

    plt.figure(figsize=(20,12))
    plt.scatter(r_samples[:,0], r_samples[:,1], label='true dist')
    plt.scatter(samples[:,0], samples[:,1], c='r', label='GAN')
    plt.legend()
    plt.savefig(model_dir + '.png',bbox_inches='tight')
    plt.close()

    return model


# In[ ]:


vanilla = make_model('model_out/vanilla2', False)

reconstruct = make_model('model_out/reconstruction_added', True)


# In[4]:


# reconstruct2 = make_model('model_out/reconstruction_added', True)

