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
# XML manipulation
import xml.etree.cElementTree as ET
# Regular expressions
import re

# Database
from .mongo import *
# Utils
from .utils import *

class images2Dataset:
    def __init__(self):
        pass

    def addData(self,
                dbFolder = os.getcwd(),
                create = False,
                db = False,
                imagesSize = "constant"):
        """
        :param dbFolder: string that contains the folder where all the
                            images live
        :param create: bool that decides whether to create a new db or not
        :param db: bool that decides to use db or not
        :param imagesSize: string that decides to use constant or multiple
                            sizes for images. Not constant parameter
                            requires padding feature.
        """
        # Set database folder
        assert dbFolder != os.getcwd(), "dbFolder can't be the same directory"
        assert type(dbFolder) == str, "dbFolder must be a string"
        assert isFolder(dbFolder) == True, "folder does not exist"
        self.dbFolder = dbFolder
        # Set subfolders
        self.subfolders = getFolders(self.dbFolder)
        #print(getFolders(self.dbFolder))
        # Set images per folder
        self.images = {}
        # Check for single folder
        if len(self.subfolders) == 0:
            self.images = getImages(dbFolder)
        else:
            for subfolder in self.subfolders:
                self.images[subfolder] = getImages(subfolder)
        # Dummy
        self.height = 100
        self.width = 100
        self.depth = 3

    def uris2MongoDB(self):
        """
        Images stored on hard disk, links available in mongoDB database
        : return: boolean that indicates success 
        """ 
        assert self.db == True,\
                "mongodb can't be used unless db parameter is set to true"
        # Get keys 
        keys = getDictKeys(self.images)
        # Save vectors inside keys to db
        for key in keys:
            # Get images of class "key"
            assert key != None,\
                    "There was a problem with key"
            imgs = self.images.get(key)
            # Write uris
            for img in imgs:
                self.dbClient.writeData(img)

    def uris2Dataframe(self, 
                        returnTo = False):
        """
        :param returnTo: bool that controls whether to return the dataframe 
                        or not
        Convert image uris to dataframe
        """
        # Check the classes have the same length
        self.images = fillDictRows(self.images)
        # Create dataframe
        self.df = pd.DataFrame(self.images)
        if returnTo:
            return self.df
        else:
            pass

    def uris2xmlAnnotations(self,
                            folderPath = os.getcwd(),
                            VOCFormat = True):
        """
        WARNING:
            * Supports a single folder with images inside
        Creates an annotation for each of the images in the input dataframe
        :param folderpath: input os path that contains the path to the images folder
        :param VOCFormat: input bool that decides if the annotations will have
                            the VOC Dataset format
        """
        # Create folder named annotations
        folderPathAnnotations = os.path.join(os.getcwd(), "annotations")
        createFolder(folderPathAnnotations)
        # Read images inside the folder
        imgsFolder = getImages(folderPath)
        # Iterate over imgs
        for imgPath in imgsFolder:
            # Read images
            frame = cv2.imread(imgPath)
            #print(imgPath, frame)
            # Get shape
            height, width, depth = frame.shape
            # Get filename
            dir_, filename = os.path.split(imgPath)
            # Get name
            match = re.match(r"([A-Za-z_]+)(_[0-9_]+)(\.jpg)", filename)
            name = match.groups()[0]
            # Convert data to VOC
            self.VOCFormat(folderPathAnnotations = folderPathAnnotations,\
                            folder = "annotations",\
                            filename = filename,\
                            path = imgPath,\
                            database = "SPID",\
                            width = str(width),\
                            height = str(height),\
                            depth = str(depth),\
                            name = name,\
                            xmin = 1,\
                            xmax = str(width-1),\
                            ymin = 1,\
                            ymax = str(height-1))

    def VOCFormat(self,
                    folderPathAnnotations,
                    folder,
                    filename,
                    path,
                    database,
                    width,
                    height,
                    depth,
                    name,
                    xmin,
                    xmax,
                    ymin,
                    ymax):
        """
        Method that converts an image to a xml annotation format in the 
        PASCAL VOC dataset format. 
        :param folderPath: input string with the path of the folder
        :param folder: input string with the name "annotations"
        :param path: input string with the path of the image
        :param database: input string with the name of the database
        :param width: input int with the width of the image
        :param height: input int with the height of the image
        :param depth: input int with the depth of the image
        :param name: input string with the name of the class the image
                    belongs to
        :param xmin: input int with the pixel the cropping starts in the
                    width of the image
        :param xmax: input int with the pixel the cropping ends in the
                    width of the image
        :param ymin: input int with the pixel the cropping starts in the 
                    height of the image
        :param ymax: input int with the pixel the cropping ends in the 
                    height of the image
        """
        # Create annotation header
        annotation = ET.Element("annotation")
        # Image information
        ET.SubElement(annotation, "folder", verified = "yes").text = str(folder)
        ET.SubElement(annotation, "filename").text = str(filename)
        ET.SubElement(annotation, "path").text = str(path)
        # Source
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = str(database)
        # Size
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        # Segemented
        ET.SubElement(annotation, "segmented").text = "0"
        # Object
        object_ = ET.SubElement(annotation, "object")
        ET.SubElement(object_, "name").text = str(name)
        ET.SubElement(object_, "pose").text = "Unspecified"
        ET.SubElement(object_, "truncated").text = "0"
        ET.SubElement(object_, "difficult").text = "0"
        # Bound box inside object
        bndbox = ET.SubElement(object_, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
        # Write file
        tree = ET.ElementTree(annotation)
        drive, path_and_file = os.path.splitdrive(filename)
        path, file = os.path.split(path_and_file)
        file = file.split(".jpg")[0] + ".xml"
        #print(os.path.join(folderPathAnnotations, file))
        tree.write(os.path.join(folderPathAnnotations, file))

    ##################################TO FIX#############################################
    def images2Tensor(self):
        """
        Convert images in each subfolder to a tensor X and a 
        tensor label Y
        : return: a tensor X of features and a tensor Y of labels
        """
        # Assert memory constraints
        assert (self.height <= 50) and (self.width <= 50),\
                RESIZE_DATASET
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
        labels = np.zeros([numberClasses,])
        # Read images in each subfolder
        for k in tqdm(range(len(keys))):
            # Get images of class "key"
            imgs = self.images.get(keys[k])
            # Read image
            for img in imgs:
                if type(img) == str:
                    frame  = cv2.imread(img).reshape(-1, 1)
                    features = np.c_[features, frame]
                    labels = np.c_[labels, classes[k].reshape(-1, 1)]
                else:
                    pass
        return features[:, 1:], labels[:, 1:]

    def images2CSV(self):
        """
        Convert images in each subfolder to vectors written in a csv file 
        : return: a tensor X of features and a tensor Y of labels
        """
        # Assert memory constraints
        assert (self.height <= 100) and (self.width <= 100),\
                SCALE_DATASET
        # Create file
        NAME_FILE = "dbResized/dataset.csv"
        file = open(NAME_FILE, "w")
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
            imgs = self.images.get(keys[k])
            # Iterate over images
            for img in imgs:
                if type(img) == str:
                    frame = cv2.imread(img).reshape(-1, 1)
                    for i in range(frame.shape[0]):
                        # Write pixels
                        file.write(str(frame[i, :]) + ",")
                    # Write labels
                    file.write(str(classes[k])+"\n")
                else:
                    pass
        # Close file
        file.close()
    ##########################################################################
