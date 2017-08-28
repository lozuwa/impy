"""
package: Images2Dataset
class: utils
Author: Rodrigo Loza
Description: Utils methods 
"""
# General purpose 
import os
import sys
# Tensor manipulation
import numpy as np

# Global variables
RESIZE_DATASET = "Your images are too big, try to scale your data"
PROBLEM_CREATING_FOLDER = "There was a problem creating the file"
DATAFRAME_IS_NONE = "You have to convert your image dataset to a dataframe first"
VECTORS_MUST_BE_OF_EQUAL_SHAPE = "Both vectors should have the same len"
RESIZING_COMPLETE = "Resize operation is complete"
RBG2GRAY_COMPLETE = "Conversion operation from RGB to GRAY complete"

def getFolders(folder):
    """
    :param folder: string that contains the name of the folder that we want to extract subfolders
    : return: list of subfolders in the folder
    """
    return [str(folder+"/"+each) for each in os.listdir(folder)]

def getImages(folder):
    """ 
    :param folder: string that contains the name of the folder that we want to extract images
    : return: list of images in the folder
    """
    return [str(folder+"/"+each) for each in os.listdir(folder)]

def isFile(file):
    """
    :param file: string that contains the name of the file we want to test
    : return: boolean that asserts if the file exists 
    """
    if os.path.isfile(file):
        return True
    else:
        return False

def isFolder(folder):
    """
    :param folder: string that contains the name of the folder we want to test
    : return: boolean that asserts if the folder exists 
    """
    if os.path.isdir(folder):
        return True
    else:
        return False

def createFolder(folder, verbosity = False):
    """
    :param folder: string that contains the name of the folder to be created.
                    It is assumed folder contains the complete path
    : return: boolean that asserts the creation of the folder 
    """
    if isFolder(folder):
        if verbosity:
            print("Folder {} already exists".format(folder))
        return True
    else:
        try:
            os.mkdir(folder)
        except:
            raise ValueError("The folder could not be created")
            return False
        return True

def getDictKeys(dict_):
    """
    :param dict_: dictionary that contains the classes and the images
    : return: dictionary's keys
    """
    keys = dict_.keys()
    return keys

def getDictValues(dict_, 
                    key):
    """
    :param dict: dictionary that contains the classes and the images
    :param key: string value that selects a specific class in dict
    : return: values for dict[key]
    """
    values = dict_.get(key, None)
    #assert type(values) == list, "Values is not a list"
    return values

def fillDictRows(dict_):
    """
    Fill missing data points so all the values in the dictionary 
    have the same length
    :param dict_: dictionary that has the keys and values to fix
    : return: return the filled dictionary 
    """
    keys = getDictKeys(dict_)
    size_ = []
    for key in keys:
        size_.append(len(dict_.get(key, None)))
    size_len = len(set(size_))
    if size_len > 1:
        print("Classes are not of the same size, fixing ...")
        # find the maximum
        max_rows = max(size_)
        # Fill the rest of the classes
        for key in keys:
            # If the class has less examples than the maximum, 
            # fill them 
            size_class = len(dict_.get(key, None))
            if size_class < max_rows:
                for i in range(max_rows - size_class):
                    dict_[key].append(np.nan)
    return dict_