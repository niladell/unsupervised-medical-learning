# Unsupervised medical feature learning

[![Github Issues](https://img.shields.io/github/issues/niladell/unsupervised-medical-learning.svg)](https://github.com/niladell/unsupervised-medical-learning/issues) 
![Model](https://img.shields.io/badge/Model%20on%20TPU-passing-green.svg)
![Data](https://img.shields.io/badge/Dataloader%20for%20TPU-passing-green.svg)

Project done in context of the Deep Learning course of ETH Zürich.

## Motivation
Machine learning is rapidly making its way into the field of medicine, most prominently into subfields such as oncology, 
pathology, radiology and genetics. Here, the vast majority of the work done so far focuses on supervised detection of 
end diagnoses. 
Despite the usefulness of this approach, we believe that, for certain clinical problems, such as complex prognostic 
predictions and treatment decisions, it is important to learn meaningful representations of the data and to 
automatically identify and extract the features that characterize certain diseases. Such views blend in with the 
growing concern regarding the inability of humans to understand the models used. The need for "interpretability" is, 
as a result, frequently invoked in these discussions.
In this project, we attempt to tackle the issue of meaningful representation learning, without supervision, in a 
medical environment. In particular, we constrain our project to the neuroimaging domain. To do so, we take an 
adversarial representation learning approach. Indeed, we were inspired by previous work on latent variable generative 
models, namely _Generative adversarial networks_ (GANs) and _Variational Autoencoders_ (VAEs), as well as more novel 
approaches where _mutual information_ (MI) between the latent (or feature) space and the data is introduced as an 
extra building block.

## How to run our code

Please consult the package requirements under the ```requirements.txt``` file.

The main flags to run our code are contained within the ```main.py``` file.
An example command to run a model would be:
```bash
python src/main.py --model_dir=gs://[BUCKET_NAME]/cifar10/outputs --data_dir=gs://[BUCKET_NAME]/cifar10/data  --tpu=[TPU_NAME] --dataset=[DATASET]
```

In order to run the project using TPU, the data needs to be loaded onto a [GCP bucket](https://cloud.google.com/storage/docs/creating-buckets). 

The results can be visualized on [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard). In GCP virtual machines they can be seen in the local machine by forwarding the ssh port, e.g.

```ssh -L 6006:localhost:6006 [HOST]```

Then, Tensorboard can be used normally from the VM, even with a GCP bucket as a target directory.

## How to download the datasets
To run the code, you will have to download the datasets.
* MNIST is readily available in the Python scikit-learn software package and will be loaded 
for you in the relevant script.
* The celebA can be found under: https://www.kaggle.com/jessicali9530/celeba-dataset
* The CQ500 dataset can be downloaded from : http://headctstudy.qure.ai/dataset


## Project structure

Below is an overview of the structure of this project:

```bash
project
│   README.md
│   requirements.txt
│   to_run.sh
│   monitor.sh
│
├───src
│   │   main.py
│   │   main_search.py
│   │
│   ├─── models
│   │    │   basic_model.py
│   │    │   resBlocks_model.py
│   │    │   resBlocks_ops.py
│   │
│   │
│   ├─── datamanager
│   │    │   celebA_input_functions.py
│   │    │   CIFAR_input_functions.py
│   │    │   cq500_input_functions.py
│   │    │   cq500_256_input_functions.py
│   │
│   │
│   ├─── core
│   │    │   core_model_estimator.py
│   │
│   │ 
│   │─── util 
│        │   cq500_folder_structure.py
│        │   dcm_manipulation.py
│        │   get_dcms_for_PCA.py
│        │   image_postprocessing.py
│        │   image_preprocessing.py
│        │   pca_mnist.py
│        │   tensorboard_logging.py
│        │   tpu_teraflops_measure.py
│        │   windowing.py
│
│
│
└───scripts
    │ compare_z-locs.py
    │ compare_z-locs_updated.py
    │ convert_dcm_to_unique_tfRecord.py
    │ ct_test.dcm
    │ fractures.csv
    │ fractures.txt
    │ generate_cifar10_tfrecords.py
    │ healthy.csv
    │ healthy.txt
    │ hemorrhage.txt
    │ hemorrhages.csv
    │ reads.csv
    │ restart_tpu.sh
    │ retrieve_all_dcms.py
    │ rewrite_normalize_dcm.py
    │ validation_dataset_creation.py
    
    
```

The ```src/``` folder contains the most important scripts to be able to run the models,
and includes the models themselves, input functions for each dataset and utility functions. 
CIFAR is included as we ran early tests on this dataset, however, we report our results 
on the MNIST, celebA and CQ500 datasets.

The ```scripts/``` folder contains a series of scripts which were necessary to analyze and
manage the data. The CQ500 dataset in particular required a lot of data analysis and special
handling, given that the original data is in a DICOM format.

## Happy model running!
Spotted an error? Feedback and pull requests are welcome!


##The authors
Nil Adell Mill  
Inês Borges Pereira  
Patrick Haller  
Lama Saouma  


&nbsp; 

&nbsp;

###### _Project using [Tensorflow Project Template](https://github.com/niladell/tensorflow-project-template). Check it for specifics about the structure and template files._
