"""Precise model definition """

# from .CIFAR_example_model import ExampleModel
from .CIFAR_example_model import ExampleModel
from .basic_model import BasicModel
from .resBlocks_model import ResModel
__all__ = (
    'ExampleModel',
    'BasicModel',
    'ResModel'
)
