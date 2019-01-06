# Unsupervised medical feature learning

[![Github Issues](https://img.shields.io/github/issues/niladell/unsupervised-medical-learning.svg)](https://github.com/niladell/unsupervised-medical-learning/issues) 
![Model](https://img.shields.io/badge/Model%20on%20TPU-passing-green.svg)
![Data](https://img.shields.io/badge/Dataloader%20for%20TPU-passing-green.svg)

Project done in context of the Deep Learning course of ETH Zürich.

In order to run the project using TPU, the data needs to be loaded onto a [GCP bucket](https://cloud.google.com/storage/docs/creating-buckets). An example of training a GAN model on CIFAR-10 can be found under `model/` and `datamanager/`. 
> python src/main.py --model_dir=gs://[BUCKET_NAME]/cifar10/outputs --data_dir=gs://[BUCKET_NAME]/cifar10/data  --tpu=[TPU_NAME]

To download the CIFAR-10 dataseet run [`util/generate_cifar10_tfrecords.py`](https://github.com/niladell/unsupervised-medical-learning/blob/master/src/util/generate_cifar10_tfrecords.py).

The results can be visualized on [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard). In GCP virtual machines they can be seen in the local machine by forwarding the ssh port, e.g.

> ssh -L 6006:localhost:6006 [HOST]

then Tensorboard can be used normally from the VM, even with a GCP bucket as a target directory.

&nbsp; 

&nbsp;

###### _Project using [Tensorflow Project Template](https://github.com/niladell/tensorflow-project-template). Check it for specifics about the structure and template files._
