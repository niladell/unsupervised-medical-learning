
# coding: utf-8

# In[1]:


import sys
sys.path.append("src")


import random

import logging
logging.basicConfig(format='%(filename)s: '
                           '%(levelname)s: '
                           '%(funcName)s(): \t'
#                            '%(lineno)d:\t'
                           '%(message)s')
from absl import flags
import tensorflow as tf

log = logging.getLogger('tensorflow')
log.setLevel('INFO')

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

# In[2]:


def make_model(model_dir, reconstruction_loss, use_encoder, encoder, number_of_gaussians, ax):
    model_dir = '{}/NG_{}'.format(model_dir, number_of_gaussians)

    log.setLevel('INFO')
    data_dir = ''
    dataset = 'TOY'
    noise_dim= 1

    # GENERIC PARAMS
    batch_size= 64
    d_optimizer= 'ADAM'

#     reconstruction_loss= True
#     use_encoder= True

    e_loss_lambda= 0.0
    e_optimizer= 'ADAM'
#     encoder= 'ATTACHED'
    eval_loss= True
    g_optimizer= 'ADAM'
    soft_label = 0

    window_lambda = 0
    use_wgan = 0
    gcp_project= None
    iterations_per_loop= 500
    lambda_window= 0
    learning_rate= 0.0002
    noise_cov= 'IDENTITY'
    num_eval_images= 1024
    num_shards= None
    num_viz_images= 100
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


#     try:
#         import shutil
#         shutil.rmtree(model_dir)
#     except FileNotFoundError:
#         print('Folder {model_dir} does not exist')

    ##### START
    model = Model(number_of_gaussians,
                model_dir=model_dir, data_dir=data_dir, dataset=dataset,
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
    model.train(8000, generate_input_fn)
    log.setLevel('ERROR')



    def get_samples(sz):
        def _input_fn(params):
             return {'random_noise':
                tf.constant(z, dtype=tf.float32)} #np.random.uniform(0,1)] #,np.random.uniform(0,1)] 
                             #for i in range(batch_size)], dtype=tf.float32)}

        gen = model.est.predict(_input_fn)
        out = [next(gen)['generated_images'] for z in range(len(sz))]
        return out

    z = np.array([[np.random.uniform(np.zeros(noise_dim),1)] for i in range(5000)])
    samples = np.vstack(get_samples(z))
    z = np.squeeze(z)


    def sample_toy_distr():
        x = np.random.normal(0, 0.05)
        y = np.random.normal(0, 0.05)
        # centers = [(0,-1), (1,0), (-0.5,0.5)]
        centers = np.vstack([[i,j] for i in range(number_of_gaussians) for j in range(number_of_gaussians)])
        centers = centers - np.mean(centers, axis=0)

        center = np.random.randint(0, centers.shape[0])
        mu_x, mu_y = centers[center]
        return [x + mu_x, y + mu_y]

    r_samples = [sample_toy_distr() for i in range(10000)]
    r_samples = np.vstack(r_samples)

    plt.figure(figsize=(20,12))
    plt.scatter(r_samples[:,0], r_samples[:,1], label='true dist')
    plt.scatter(samples[:,0], samples[:,1], c=z, label='GAN')
    plt.legend()
    plt.savefig(model_dir + 'final_output.png')

    ax.scatter(r_samples[:,0], r_samples[:,1], label='true dist')
    ax.scatter(samples[:,0], samples[:,1], c=z, label='GAN')
    # ax.legend()

    return model


# In[3]:

gaussians_list = [1, 4, 20]
fig = plt.figure(figsize=(20,12))
cols = 5
for i in range(3):
    gs = gaussians_list[i]
    ax1 = fig.add_subplot(3,cols,i*cols+1)
    ax2 = fig.add_subplot(3,cols,i*cols+2)
    ax3 = fig.add_subplot(3,cols,i*cols+3)
    ax4 = fig.add_subplot(3,cols,i*cols+4)
    ax5 = fig.add_subplot(3,cols,i*cols+5)


    vanilla = make_model('model_out/vanilla', False, False, 'ATTACHED', gs, ax1)
    encoder = make_model('model_out/encoder_attached', False, True, 'ATTACHED', gs, ax2)
    encoder_i = make_model('model_out/encoder_indep', False, True, 'INDEPENDENT', gs, ax3)
    reconstruct = make_model('model_out/reconstruction_attached', True, True, 'ATTACHED', gs, ax4)
    reconstruct_i = make_model('model_out/reconstruction_indep', True, True, 'INDEPENDENT', gs, ax5)

fig.savefig('model_out/final_output.png')

# def do_all_plots(*kargs):

#     fig = plt.figure()
#     ax1 = fig.add_subplot(211)
#     ax2 = fig.add_subplot(212)
#     make_plot(model, number_of_gaussians)

# # TODO Clean and put repeated functions toguether
# def make_plot(model, number_of_gaussians):
#     def get_sample(batch_size, z):
#         def _input_fn(params):
#              return {'random_noise':
#                 tf.constant(z, dtype=tf.float32)} #np.random.uniform(0,1)] #,np.random.uniform(0,1)] 
#                                  #for i in range(batch_size)], dtype=tf.float32)}

#         return next(model.est.predict(_input_fn))['generated_images']

#     z = [np.random.uniform(0,1) for i in range(100*64)]
#     samples = [get_sample(64, z[i*64:(i+1)*64]) for i in range(100)]
#     samples = np.vstack(samples)


#     def sample_toy_distr():
#         x = np.random.normal(0, 0.05)
#         y = np.random.normal(0, 0.05)
#         # centers = [(0,-1), (1,0), (-0.5,0.5)]
#         centers = np.vstack([[i,j] for i in range(number_of_gaussians) for j in range(number_of_gaussians)])
#         centers = centers - np.mean(centers, axis=0)

#         center = np.random.randint(0, number_of_gaussians)
#         mu_x, mu_y = centers[center]
#         return [x + mu_x, y + mu_y]

#     r_samples = [sample_toy_distr() for i in tqdm(np.linspace(-1, 1,10000))]
#     r_samples = np.vstack(samples)

#     plt.figure(figsize=(20,12))
#     plt.scatter(r_samples[:,0], r_samples[:,1], label='true dist')
#     plt.scatter(samples[:,0], samples[:,1], c=z, label='GAN')
#     plt.legend()
