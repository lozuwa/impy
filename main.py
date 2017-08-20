"""
package: Images2Dataset
class: Images2Dataset (main)
Author: Rodrigo Loza
Description: This library is intended to convert an image raw dataset to a vectorial structured dataset.
The library assumes:
* Mongodb is running
* The folder structure follows:
                                - Db Folder
                                    - Dataset 0
                                    - Dataset 1
                                    - ...
* All the images have the same size 
"""
# General purpose 
import os
import sys
from tqdm import tqdm
# Matrix manipulation
import numpy as np 
# Data manipulation
import pandas as pd
# Image manipulation
import cv2
from PIL import Image

# Database 
from mongo import *
# Utils
from utils import *
# Stats
from stats import *

### "C:/Users/HP/Dropbox/pfm/Databases/TEST/"

class Images2Dataset:
    def __init__(self, 
                dbFolder = os.getcwd(), 
                create = False, 
                db = False, 
                imagesSize = "constant"):
        """
        :param dbFolder: string that contains the folder where all the images live
        :param create: bool that decides whether to create a new db or not
        :param db: bool that decides to use db or not
        :param imagesSize: string that decides to use constant or multiple sizes for images
                            Not constant parameter requires padding feature. 
        """ 
        # Set database folder
        assert dbFolder != os.getcwd(), "dbFolder can't be the same directory"
        assert type(dbFolder) == str, "dbFolder must be a string"
        assert isFolder(dbFolder) == True, "folder does not exist"
        self.dbFolder = dbFolder
        # Set subfolders
        self.subfolders = getFolders(self.dbFolder)
        # Set images per folder
        self.images = {}
        for subfolder in self.subfolders:
            self.images[subfolder] = getImages(subfolder)
        # Instantiate db client
        self.db = db
        if db:
            self.dbClient = MongoDb(name = "data", create = create)
        else:
            pass
        # If imagesSize is constant, then get a sample
        if imagesSize == "constant":
            self.height, self.width, self.depth = cv2.imread(self.images[self.subfolders[0]][0]).shape
        # Empty dataframe
        self.df = None

    def stats():
        eda = EDA(self.images)
        eda.descriptiveStats(totalPixels = self.height*self.width*self.depth)

    def uris2MongoDB(self):
        """
        Images stored on hard disk, links available in mongoDB database
        : return: boolean that indicates success 
        """ 
        assert self.db == True, "mongodb can't be used unless db parameter is set to true"
        # Get keys 
        keys = getDictKeys(self.images)
        # Save vectors inside keys to db
        for key in keys:
            # Get images of class "key"
            assert key != None, "There was a problem with key"
            imgs = self.images.get(key)
            # Write uris
            for img in imgs:
                self.dbClient.writeData(img)

    def uris2Dataframe(self):
        """
        Convert image uris to dataframe
        : return: pandas dataframe that contains the classes as columns
                  and the uris of each class as rows 
        """ 
        # Check the classes have the same length 
        keys = getDictKeys(self.images)
        size_ = []
        for key in keys:
            size_.append(len(self.images.get(key, None)))
        size_len = len(set(size_))
        if size_len > 1:
            print("Classes are not of the same size")
            # find the maximum
            max_rows = max(size_)
            # Fill the rest of the classes
            for key in keys:
                # If the class has less examples than the maximum, fill them 
                size_class = len(self.images.get(key, None))
                if size_class < max_rows:
                    for i in range(max_rows - size_class):
                        self.images[key].append(np.nan)
        # Create dataframe
        self.df = pd.DataFrame(self.images)
        return self.df

    ##################################TO FIX#############################################
    def images2Tensor(self):
        """
        Convert images in each subfolder to a tensor X and a 
        tensor label Y
        : return: a tensor X of features and a tensor Y of labels
        """
        # Assert memory constraints
        assert (self.height < 50) and (self.width < 50), SCALE_DATASET
        # Previous calculations
        rows = self.height*self.width*self.depth
        # Get keys
        keys = [each for each in self.images.keys()]
        columns = []
        for key in keys:
            columns.append(len(self.images.get(key)))
        # Number of classes
        numberClasses = len(self.subfolders)
        # Data (rows are the number of features or pixels) 
        # (columns are the number of examples)
        features = np.zeros([rows,])
        # Labels
        classes = np.eye(numberClasses)
        labels = np.zeros[numberClasses,]
        # Read images in each subfolder
        for k in tqdm(range(len(keys))):
            # Get images of class "key"
            assert keys[k] != None, "There was a problem with key"
            imgs = self.images.get(keys[k])
            # Read image 
            for img in tqdm(imgs):
                features = np.c_[features, cv2.imread(img).reshape(-1, 1)]
                labels = np.c_[labels, classes[k].reshape(-1, 1)]
        return features, labels

    def images2CSV(self):
        """
        Conver images in each subfolder to vectors written in a csv file 
        : return: a tensor X of features and a tensor Y of labels
        """
        # Assert memory constraints
        assert (self.height < 100) and (self.width < 100), SCALE_DATASET
        # Create file
        file = open("dataset.csv", "w")
        # Get keys
        keys = [each for each in self.images.keys()]
        # Number of classes
        numberClasses = len(keys)
        classes = [each for each in range(numberClasses)]
        # Write columns
        for each in range(self.height*self.width*self.depth):
            file.write(str(each)+",")
        file.write("label"+"\n")
        # Read images in each subfolder
        for k in tqdm(range(len(keys))):
            # Get images of class "key"
            assert keys[k] != None, "There was a problem with key"
            imgs = self.images.get(keys[k])
            # Iterate over images
            for img in tqdm(imgs):
                frame = cv2.imread(img).reshape(-1, 1)
                for i in range(frame.shape[0]):
                    # Write pixels
                    file.write(str(frame[i, :]) + ",")
                # Write labels
                file.write(str(classes[k])+"\n")
        # Close file
        file.close()
    ##########################################################################