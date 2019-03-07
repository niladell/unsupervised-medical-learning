
# coding: utf-8

# In[1]:



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


def make_model(model_dir, reconstruction_loss, use_encoder, encoder, ax):

    log.setLevel('INFO')
    data_dir = ''
    dataset = 'dSprites'
    noise_dim= 6

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

    from model import BasicModel as Model
    from datamanager.dSprites_input_functions import generate_input_fn


#     try:
#         import shutil
#         shutil.rmtree(model_dir)
#     except FileNotFoundError:
#         print('Folder {model_dir} does not exist')

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
    log.setLevel('ERROR')
    print('Done training')

    def get_samples(sz):
        def _input_fn(params):
             return {'random_noise':
                tf.constant(z, dtype=tf.float32)} #np.random.uniform(0,1)] #,np.random.uniform(0,1)] 
                                 #for i in range(batch_size)], dtype=tf.float32)}

        
        out = [next(model.est.predict(_input_fn))['generated_images'] for i in range(len(z))]
        return out
    
    base_z = np.zeros((1, noise_dim))
    samples = []
    for i in range(noise_dim):
        print(i)
        z = np.repeat(base_z, repeats=8, axis=0)
        for j, val in enumerate(np.linspace(-1.5, 1.5, 8)):
            z[j, i] = val
        tmp = np.concatenate(get_samples(z), 1)
        samples.append(tmp)
    samples = np.concatenate(samples, axis=0)
    print(samples)
    print(samples.shape)
    # tf.logging.error('z: {} sample:{}'.format(z, samples))


    # r_samples = [sample_toy_distr() for i in range(10000)]
    # r_samples = np.vstack(r_samples)

    if len(samples.shape) == 3:
        samples = np.squeeze(samples)

    plt.figure(figsize=(20,12))
    plt.imshow(samples)
    plt.savefig(model_dir + 'final_output.png')

    ax.imshow(samples)
    # ax.legend()

    return model


# In[ ]:



# gaussians_list = [1, 4, 20]
fig = plt.figure(figsize=(20,12))
cols = 5
ax1 = fig.add_subplot(1,cols,1)
ax2 = fig.add_subplot(1,cols,2)
ax3 = fig.add_subplot(1,cols,3)
ax4 = fig.add_subplot(1,cols,4)
ax5 = fig.add_subplot(1,cols,5)

vanilla = make_model('model_dSprites/vanilla', False, False, 'ATTACHED', ax1)
encoder = make_model('model_dSprites/encoder_attached', False, True, 'ATTACHED', ax2)
encoder_i = make_model('model_dSprites/encoder_indep', False, True, 'INDEPENDENT', ax3)
reconstruct = make_model('model_dSprites/reconstruction_attached', True, True, 'ATTACHED', ax4)
reconstruct_i = make_model('model_dSprites/reconstruction_indep', True, True, 'INDEPENDENT', ax5)

fig.savefig('model_dSprites/final_output.png')
