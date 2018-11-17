# Tensorflow project template

![Work being carried out](https://img.shields.io/badge/%20functional%20code-minimal%20functionallity-yellow.svg)

This is the base template I use when starting a Tensorflow project. The motivation of having such a thing is to have all the more cumbersome part done under the hood and allow myself to focus on the actual architecture and the specific loading and preprocessing of my dataset.

In addition this allows me to have a much modular approach to projects, where if for instance, I want to try a model in a new dataset I just need to re-adjust the data manager without having to focus on the rest of the elements.

## Basic elements

Ideally just three elements of this template should be important for starting a new project:

| | Sumarized:| |
|---|---|---|
| `main.py`    | Main runfile, here we load the model and the data manager and run the network. Also all parameters (learning rate, optimizer, num. epochs...) will be defined here. |
| `models/` [1] | The architecture definition will be done here in a class based on the code model, inside `define_net` (there is an example model of a CNN network for classification there) |
| `datamanager/` | The data managing object will be defined here, i.e. where to get the data, if to pull from different files, which pre-processing is necessary, any data augmentation techinques, etc. (this project contains an example for the CIFAR-10 dataset) |

> **[1]**  In particular the model network should get as input an iterator element from our data manager (which basically means that it should start with something like)
> ```python
>    class MyNetwork(CoreModel):
>        def define_net(self, inputs):
>            my_data, my_labels = inputs
>                ...
>                // NETWORK DEFINITION
>                ...
>            return [predictions], [loss], [others]
>    ```

## *[Documentation in progress]*

***

## Project structure

The project is structured in the following fashion:
```bash
project
│   README.md
│
├───src
│   │   main.py
│   │
│   ├─── models
│   │    │   model_class1.py
│   │    │   model_class2.py
│   │    │   ...
│   │
│   ├─── datamanager
│   │    │   data_class1.py
│   │    │   data_class2.py
│   │    │   ...
│   │
│   ├─── core
│   │    │   core_model.py
│   │    │   core_datamanager.py
│
│
└───datasets
    ├─── dataset_1
    ├───  ....
    │
```