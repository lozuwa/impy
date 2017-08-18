"""
package: Images2Dataset
class: utils
Author: Rodrigo Loza
Description: Utils methods 
"""
# General purpose 
import os
import sys

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

def getDictKeys(dict):
    """
    :param dict: dictionary that contains the classes and the images
    : return: dictionary's keys
    """
    keys = dict.keys()
    return keys

def getDictValues(dict, key):
    """
    :param dict: dictionary that contains the classes and the images
    :param key: string value that selects a specific class in dict
    : return: values for dict[key]
    """
    values = dict.get(key, None)
    assert values == list, "Values is not a list"
    return values
