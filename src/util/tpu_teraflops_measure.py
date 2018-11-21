"""Simple scritpt from google TPU collab to measure teraflops
   [https://colab.research.google.com/notebooks/tpu.ipynb]
"""

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver #pylint: disable=E0611
import time
import tensorflow as tf


#tpu_address = ['node-1', 'node-2']
# Apparently multiple TPUs for a single session are not
# yet suported
tpu_address = ['node-1']

tpu_cluster = TPUClusterResolver(
                    tpu=tpu_address
                ).get_master()

N = 4096
COUNT = 100

def flops():
  x = tf.random_uniform([N, N])
  y = tf.random_uniform([N, N])
  def _matmul(x, y):
    return tf.tensordot(x, y, axes=[[1], [0]]), y

  return tf.reduce_sum(
    tpu.repeat(COUNT, _matmul, [x, y])
  )

tpu_ops = tpu.batch_parallel(flops, [], num_shards=8)

session = tf.Session(tpu_cluster)

try:
  print('Warming up...')
  session.run(tpu.initialize_system())
  session.run(tpu_ops)
  print('Profiling')
  start = time.time()
  session.run(tpu_ops)
  end = time.time()
  elapsed = end - start
  print(elapsed, 'TFlops: {:.2f}'.format(1e-12 * 8 * COUNT * 2*N*N*N / elapsed))
except Exception as e:
    print(e)
finally:
  session.run(tpu.shutdown_system())
  session.close()