"""
package: Images2Dataset
class: unitTests
Author: Rodrigo Loza
Description: Unit tests
"""
# General purpose
import os
import sys
import tqdm
# Data manipulation
import pandas as pd
# Tensor manipulation
import numpy as np
# Image manipulation
import cv2
from PIL import Image
import PIL
# Visualization
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
# Image
import cv2
import PIL
# Main
from .images2Dataset import images2Dataset as im2da
# Utils
from .utils import *
# Stats
from .stats import *
# Preprocess
from .preprocess import *

DB_FOLDER = os.path.join(os.getcwd(), "tests", "db")
DBRESIZED_FOLDER = os.path.join(os.getcwd(), "dbResized")

################ SINGLE METHODS #######################
def test_fillDictRows():
    dict_ = {'a': [1,2,3], 'b': [1,2]}
    dict_ = fillDictRows(dict_)
    print(dict_)

def test_uris2Dataframe():
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    df = dataset.uris2Dataframe(returnTo = True)
    print(df)

def test_tensorSizes():
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    df = dataset.uris2Dataframe(returnTo = True)
    stats = stats(df)
    stats.tensorSizes()

def test_classesBalance():
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    df = dataset.uris2Dataframe(returnTo = True)
    stats_ = stats(df)
    stats_.classesBalance(visualize = True)

def test_resizeImages():
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    df = dataset.uris2Dataframe(returnTo = True)
    stats_ = stats(df)
    stats_.tensorSizes()
    prep = preprocessImageDataset(df)
    prep.resizeImages(width = 300, height = 300)

def test_rgb2gray():
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    df = dataset.uris2Dataframe(returnTo = True)
    prep = preprocessImageDataset(df)
    prep.rgb2gray()

def test_splitImageDataset():
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    df = dataset.uris2Dataframe(returnTo = True)
    prep = preprocessImageDataset(df)
    trainImgsPaths, testImgsPaths, trainImgsClass, testImgsClass = prep.splitImageDataset()

def test_divideIntoPatches():
    NAME_IMG = os.getcwd()+"/tests/img.jpg"
    frame = cv2.imread(NAME_IMG)
    imageHeight, imageWidth, depth = frame.shape
    slideWindowSize = (350,350)
    strideSize = (350,350)
    padding = "VALID"
    prep = preprocessImage()
    patchesCoordinates, numberPatchesHeight,\
            numberPatchesWidth = prep.divideIntoPatches(imageWidth,
                                                        imageHeight,
                                                        slideWindowSize,
                                                        strideSize,
                                                        padding)
    frame = drawGrid(frame.copy(),\
                    patchesCoordinates,\
                    [1 for each in patchesCoordinates])
    cv2.namedWindow("adf", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("adf", 640, 530)
    cv2.imshow("adf", frame)
    cv2.waitKey(0)

def test_divideIntoPatchesSAMEPADDING():
    WINDOW_SIZE = 700
    NAME_IMG = os.getcwd()+"/tests/img.jpg"
    frame = cv2.imread(NAME_IMG)
    imageHeight, imageWidth, depth = frame.shape
    slideWindowSize = (WINDOW_SIZE, WINDOW_SIZE)
    strideSize = (WINDOW_SIZE, WINDOW_SIZE)
    padding = "SAME"
    prep = preprocessImage()
    patchesCoordinates, numberPatchesHeight,\
    numberPatchesWidth, zeros_h,\
    zeros_w = prep.divideIntoPatches(imageWidth,
                                    imageHeight,
                                    slideWindowSize,
                                    strideSize,
                                    padding)
    frame = lazySAMEpad(frame.copy(),
                        zeros_h,
                        zeros_w)
    frame = drawGrid(frame.copy(),\
                    patchesCoordinates,\
                    [1 for each in patchesCoordinates])
    cv2.namedWindow("adf", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("adf", 640, 530)
    cv2.imshow("adf", frame)
    cv2.waitKey(0)

def test_uris2xmlAnnotations():
    from impy.images2Dataset import images2Dataset as imda
    from impy.utils import *
    import pandas as pd
    import os
    
    dataset = im2da()
    path = os.path.join("C://Users//HP//Dropbox//Databases//", "ASCARIS_LUMBRICOIDES")

    df = getImages(path)
    df = pd.DataFrame({"ASCARIS": df})
    print(type(df))

    #dataset.uris2xmlAnnotations(df = df)


################ PROCESSES #######################
def test_images2Tensor():
    # Create images2Dataset instance
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    # Convert to dataframe
    df = dataset.uris2Dataframe(returnTo = True)
    # Compute stats
    stats_ = stats(df)
    stats_.tensorSizes()
    # Create preprocess instance
    prep = preprocess(df)
    # Resize data
    prep.resizeImages(width = 50, height = 50)
    # Create new Images2Dataset instance
    dataset = im2da(dbFolder = DBRESIZED_FOLDER)
    # Get tensored dataset
    features, labels = dataset.images2Tensor()
    print(features.shape, labels.shape)

def test_images2CSV():
    # Create images2Dataset instance
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    # Convert to dataframe
    df = dataset.uris2Dataframe(returnTo = True)
    # Compute stats
    stats_ = stats(df)
    stats_.tensorSizes()
    # Create preprocess instance
    prep = preprocess(df)
    # Resize data
    prep.resizeImages(width = 20, height = 20)
    # Create new Images2Dataset instance
    dataset = im2da(dbFolder = DBRESIZED_FOLDER)
    # Get tensored dataset
    dataset.images2CSV()

def test_saveImageDatasetKeras():
    dataset = im2da()
    dataset.addData(dbFolder = DB_FOLDER)
    df = dataset.uris2Dataframe(returnTo = True)
    prep = preprocessImageDataset(df)
    trainImgsPaths, trainImgsClass, testImgsPaths, testImgsClass = prep.splitImageDataset()
    #print(len(trainImgsPaths), len(trainImgsClass))
    #print(len(testImgsPaths), len(testImgsClass))
    prep.saveImageDatasetKeras(trainImgsPaths,
                          trainImgsClass,
                          testImgsPaths,
                          testImgsClass)

if __name__ == "__main__":
    # Which one would you like to test?
    test_uris2xmlAnnotations()
