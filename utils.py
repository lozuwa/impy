"""
package: Images2Dataset
class: utils
Author: Rodrigo Loza
Description: Utils methods 
"""
# General purpose 
import os
import sys

# Global variables 
SCALE_DATASET = "Your images are too big, try to scale your data"

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

def getDictKeys(dict_):
    """
    :param dict_: dictionary that contains the classes and the images
    : return: dictionary's keys
    """
    keys = dict_.keys()
    return keys

def getDictValues(dict_, key):
    """
    :param dict: dictionary that contains the classes and the images
    :param key: string value that selects a specific class in dict
    : return: values for dict[key]
    """
    values = dict_.get(key, None)
    assert values == list, "Values is not a list"
    return values
