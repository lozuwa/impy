"""
Name: Images2Dataset
Author: Rodrigo Loza
Description: This library is intended to be used with the goal of converting an image raw dataset to a vectorial dataset.
"""
import os
import sys
import numpy as np 
import pandas as pd
import cv2
from PIL import Image

class Images2Dataset:
    def __init__(self, dbFolder):
        """ 
        :param dbFolder: string that contains the folder where all the raw data lives
        """ 
        assert type(dbFolder) == str, "dbFolder must be a string"
        assert isFolder(dbFolder) == True, "folder does not exist"
        self.dbFolder = dbFolder

    def getFolders(self, folder):
        """ 
        :param folder: string that contains the name of the folder that we want to extract subfolders
        : return: list of subfolders in the folder
        """
        return os.listdir(os.getcwd()+"/"+folder)

    def getImages(self, folder):
        """ 
        :param folder: string that contains the name of the folder that we want to extract images
        : return: list of images in the folder
        """
        return os.listdir(os.getcwd()+"/"+self.dbFolder+"/"+folder+"/")

    def images2Dataframe(self):
        # Variables 
        dataset = {}
        # Get subfolders
        folders = self.getFolders(self.dbFolder)
        # Process each folder
        for folder in folders:
            # Get images of current folder
            imgs = getImages(folder)
            # Create data tensor for the current folder
            img = cv2.imread(imgs[0])
            r, c, d = img.shape
            data = np.zeros([r*c*d, len(imgs)])
            # Convert images to flat vectors and store them
            for i in range(len(imgs)):
                data[:,i] = cv2.imread(imgs[i]).reshape(-1, 1)
            # Update dictionary
            dataset[folder] = data
        # Convert data to dataframe
        df = pd.DataFrame(dataset)
        return df

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
    if os.path.isfolder(folder):
        return True
    else:
        return False