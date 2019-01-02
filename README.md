# Unsupervised medical feature learning

![Data Pipeline](https://img.shields.io/badge/Data%20Pipeline-Failing-red.svg)  ![Model](https://img.shields.io/badge/Model-Not%20implemented-lightgrey.svg)
 ![TPU Build](https://img.shields.io/badge/Build%20TPU-failing-red.svg)

Project done in context of the Deep Learning course of ETH Zürich.

In order to run the project using TPU, the data needs to be loaded onto a [GCP bucket](https://cloud.google.com/storage/docs/creating-buckets). An example of training a GAN model on CIFAR-10 can be found under `model/` and `datamanager/`. 
> python src/main.py --model_dir=gs://[BUCKET_NAME]/cifar10/outputs --data_dir=gs://[BUCKET_NAME]/cifar10/data  --tpu=[TPU_NAME]

To download the CIFAR-10 dataseet run [`util/generate_cifar10_tfrecords.py`](https://github.com/niladell/unsupervised-medical-learning/blob/master/src/util/generate_cifar10_tfrecords.py).


_Project using [Tensorflow Project Template](https://github.com/niladell/tensorflow-project-template). Check it for specifics about the structure and template files._
