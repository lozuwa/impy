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
# Local classes
"""from .images2Dataset import images2Dataset as im2da
from .utils import *
from .stats import *
from .preprocess import *"""
from impy.images2Dataset import images2Dataset as im2da
from impy.utils import *
from impy.stats import *
from impy.preprocess import *

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

def test_get_valid_padding():
	slide_window_height = 500
	stride_height = 500
	image_height = 2200
	slide_window_width = 500
	stride_width = 500
	image_width = 2200
	num_patches_height, num_patches_width = get_valid_padding(slide_window_height,
															 stride_height,
															 image_height,
															 slide_window_width,
															 stride_width,
															 image_width)
	print(num_patches_height, num_patches_width)

def test_get_same_padding():
	slide_window_height = 500
	stride_height = 500
	image_height = 2600
	slide_window_width = 500
	stride_width = 500
	image_width = 2600
	zeros_h, zeros_w = get_same_padding(slide_window_height,
															 stride_height,
															 image_height,
															 slide_window_width,
															 stride_width,
															 image_width)
	print(zeros_h, zeros_w)

def test_divideIntoPatches():
    NAME_IMG = os.getcwd()+"/tests/img.jpg"
    frame = cv2.imread(NAME_IMG)
    print("Image size: ", frame.shape)
    image_height, image_width, depth = frame.shape
    slide_window_size = (500, 500)
    stride_size = (250, 250)
    padding = "VALID"
    prep = preprocessImage()
    patches_coordinates, \
    number_patches_height, number_patches_width = prep.divideIntoPatches(image_width,
			                                                        image_height,
			                                                        slide_window_size,
			                                                        stride_size,
			                                                        padding)
    #print(patchesCoordinates)
    frame = drawGrid(frame.copy(),\
                    patches_coordinates,\
                    [1 for each in patches_coordinates])
    cv2.namedWindow("adf", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("adf", 640, 530)
    cv2.imshow("adf", frame)
    cv2.waitKey(0)

def test_divideIntoPatchesSAMEPADDING():
    WINDOW_SIZE = 500
    STRIDE_SIZE = 500
    NAME_IMG = os.getcwd()+"/tests/img.jpg"
    frame = cv2.imread(NAME_IMG)
    image_height, image_width, depth = frame.shape
    slide_window_size = (WINDOW_SIZE, WINDOW_SIZE)
    stride_size = (STRIDE_SIZE, STRIDE_SIZE)
    padding = "SAME"
    prep = preprocessImage()
    patches_coordinates, number_patches_height,\
    number_patches_width, zeros_h,\
    zeros_w = prep.divideIntoPatches(image_width,
                                    image_height,
                                    slide_window_size,
                                    stride_size,
                                    padding)
    print("Before padding: {}".format(frame.shape))
    frame = lazySAMEpad(frame.copy(),
                        zeros_h,
                        zeros_w)
    print("After padding: {}".format(frame.shape))
    frame = drawGrid(frame.copy(),\
                    patches_coordinates,\
                    [1 for each in patches_coordinates])
    cv2.namedWindow("adf", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("adf", 640, 530)
    cv2.imshow("adf", frame)
    cv2.waitKey(0)

def test_uris2xmlAnnotations():
    #from impy.images2Dataset import images2Dataset as imda
    #from impy.utils import *
    #import pandas as pd
    #import os
    
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
    test_divideIntoPatchesSAMEPADDING()
