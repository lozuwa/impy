"""
Impy
=====
A library that helps to give format to an image data set.

Provides
  1. Split your image dataset.
  2. Common image preprocessing operations: crop, pad, etc.
  3. Compute stats: class balance.

Documentation
----------------------------

The docstring examples assume that `numpy` has been imported as `np`::

  >>> from impy.Preprocess import PreprocessImage
  >>> from impy.Images2Dataset import Images2Dataset

"""
# Import libraries 
import sys

# Load modules
<<<<<<< HEAD
# try:
#     from .utils import *
# except:
#     raise ImportError("utils library could not be loaded")
# try:
#     from .preprocess import *
# except:
#     raise ImportError("preprocess libary could not be loaded")
# try:
#     from .stats import *
# except:
#     raise ImportError("stats could not be loaded")
# try:
#     from .images2Dataset import *
# except:
#     raise ImportError("stats could not be loaded")
# try:
# 	from .load_tensorflow_models import *
# except:
# 	raise ImportError("load_tensorflow_models could not be loaded")
# Information
__author__ = "Rodrigo Alejandro Loza Lucero / lozuwaucb@gmail.com"
__version__ = "0.4"
__log__ = "Removed object detection class. The paths were messy and hardcoded, feature left for the future. \
		Wrote a cleaner implementation of preprocessImage.divideIntoPatches, also fixed issues with weird patching sizes."


# Data augmentation
"""
DataAugmentationMethods -> Interface that defines the 
                            methods to be implemented.

DataAugmentation -> Implements DataAugmentationMethods
                    and implements all of the methods.

DataAugmentation_test -> Tests the methods implemented 
                          in data augmentation.

"""                    
