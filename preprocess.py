"""
package: Images2Dataset
class: imageProcess
Author: Rodrigo Loza
Description: Preprocess image dataset or singular images
"""
# General purpose
import os
from tqdm import tqdm
# Image manipulation
import cv2
import PIL
from PIL import Image
# Utils 
from .utils import *
# Tensor manipulation
import numpy as np
from numpy import r_, c_
# Model selection utils from scikit-learn
from sklearn.model_selection import train_test_split

class preprocessImageDataset:
    """
    Allows operations on images
    """
    def __init__(self,
                data = []):
        """ 
        Constructor of the preprcoessImageDataset
        :param data: pandas dataframe that contains as features the
                    name of the classes and as values the paths of 
                    the images. 
        """
        self.data = data

    def resizeImages(self,
                    width = 300,
                    height = 300):
        """
        Resizes all the images in the db. The new images are stored
        in a local folder named "dbResized"
        :param width: integer that tells the width to resize 
        :param height: integer that tells the height to resize
        """
        # Create new directory
        DB_PATH = os.getcwd() + "/dbResized/"
        assert createFolder(DB_PATH) == True,\
                PROBLEM_CREATING_FOLDER
        # Read Images
        keys = getDictKeys(self.data)
        for key in tqdm(keys):
            imgs = self.data.get(key, None)
            # Create subfolders for eanumberPatchesHeight subfolder
            NAME_SUBFOLDER = key.split("/")[-1]
            assert createFolder(DB_PATH + NAME_SUBFOLDER) == True,\
                    PROBLEM_CREATING_FOLDER
            # Iterate images 
            for img in imgs:
                # Filter nan values 
                if type(img) == str:
                    # Open image
                    frame = Image.open(img)
                    # Resize image
                    frame = frame.resize((width, height),\
                                            PIL.Image.LANCZOS)
                    IMAGE_NAME = "/" + img.split("/")[-1]
                    # Save the image 
                    frame.save(DB_PATH + NAME_SUBFOLDER + IMAGE_NAME)
                else:
                    pass
        print(RESIZING_COMPLETE)

    def rgb2gray(self):
        """
        Converts the images to grayscale. The new images are stored
        in a local folder named dbGray
        """
        # Create new directory
        DB_PATH = os.getcwd() + "/dbGray/"
        assert createFolder(DB_PATH) == True,\
                PROBLEM_CREATING_FOLDER
        # Read images 
        keys = getDictKeys(self.data)
        for key in tqdm(keys):
            imgs = self.data.get(key, None)
            # Create subfolders
            NAME_SUBFOLDER = key.split("/")[-1]
            assert createFolder(DB_PATH + NAME_SUBFOLDER) == True,\
                    PROBLEM_CREATING_FOLDER
            for img in imgs:
                # Filter nan values 
                if type(img) == str:
                    # Read image
                    frame = Image.open(img)
                    # Convert RGBA to GRAYSCALE
                    frame = frame.convert(mode = "1") #dither = PIL.FLOYDSTEINBERG)
                    # Save the image
                    IMAGE_NAME = "/" + img.split("/")[-1]
                    frame.save(DB_PATH + NAME_SUBFOLDER + IMAGE_NAME)
                else:
                    pass
        print(RBG2GRAY_COMPLETE)

    def splitImageDataset(self,
                        trainSize = 0.80,
                        validationSize = 0.20):
        """
        Splits the dataset into a training and a validation set
        :param trainSize: int that represents the size of the training set
        :param validationSize: int that represents the size of the 
                                validation set
        : return: four vectors that contain the training and test examples
                    trainImgsPaths, trainImgsClass
                    testImgsPaths, testImgsClass
        """
        # Split dataset in the way that we get the paths of the images 
        # in a train set and a test set. 
        # ----- | class0, class1, class2, class3, class4
        # TRAIN | path0   path1   path2   path3   path4
        # TRAIN | path0   path1   path2   path3   path4
        # TRAIN | path0   path1   path2   path3   path4
        # TRAIN | path0   path1   path2   path3   path4
        # TEST* | path0   path1   path2   path3   path4
        # TEST* | path0   path1   path2   path3   path4
        trainX, testX, trainY, testY = train_test_split(self.data,\
                                                        self.data,\
                                                        train_size = trainSize,\
                                                        test_size = validationSize)
        print(trainX.shape, testX.shape)
        # Once the dataset has been partitioned, we are going 
        # append the paths of the matrix into a single vector.
        # TRAIN will hold all the classes' paths in train
        # TEST will hold all the classes' paths in test
        # Each class wil have the same amount of examples.
        trainImgsPaths = []
        trainImgsClass = []
        testImgsPaths = [] 
        testImgsClass = []
        # Append TRAIN and TEST images 
        keys = self.data.keys()
        for key in keys:
            imgsTrain = trainX[key]
            imgsTest = testX[key]
            for imgTrain in imgsTrain:
                if type(imgTrain) != str:
                    pass
                else:
                    trainImgsPaths.append(imgTrain)
                    trainImgsClass.append(key)
            for imgTest in imgsTest:
                if type(imgTest) != str:
                    pass
                else:
                    testImgsPaths.append(imgTest)
                    testImgsClass.append(key)
        return trainImgsPaths,\
                trainImgsClass,\
                testImgsPaths,\
                testImgsClass

    def saveImageDatasetKeras(self,
                                trainImgsPaths,
                                trainImgsClass,
                                testImgsPaths,
                                testImgsClass):
        # Create folder
        DB_PATH = os.getcwd() + "/dbKerasFormat/"
        result = createFolder(DB_PATH)
        assert result == True,\
                PROBLEM_CREATING_FOLDER
        # Create train subfolder
        TRAIN_SUBFOLDER = "train/"
        result = createFolder(DB_PATH + TRAIN_SUBFOLDER)
        assert result == True,\
                PROBLEM_CREATING_FOLDER
        # Create validation subfolder
        VALIDATION_SUBFOLDER = "validation/"
        result = createFolder(DB_PATH + VALIDATION_SUBFOLDER)
        assert result == True,\
                PROBLEM_CREATING_FOLDER
        # Create classes folders inside train and validation 
        keys = self.data.keys()
        for key in keys:
            # Train subfolder
            NAME_SUBFOLDER = key.split("/")[-1]
            result = createFolder(DB_PATH + TRAIN_SUBFOLDER +\
                                 NAME_SUBFOLDER)
            assert result == True,\
                    PROBLEM_CREATING_FOLDER
            # Test subfolder
            NAME_SUBFOLDER = key.split("/")[-1]
            result = createFolder(DB_PATH + VALIDATION_SUBFOLDER +\
                                 NAME_SUBFOLDER)
            assert result == True,\
                    PROBLEM_CREATING_FOLDER

        ######################## OPTIMIZE ########################
        # Save train images
        # Read classes in trainImgsClass
        for i in tqdm(range(len(trainImgsClass))):
            imgClass = trainImgsClass[i].split("/")[-1]
            for key in keys:
                NAME_SUBFOLDER = key.split("/")[-1]
                #print(imgClass, NAME_SUBFOLDER)
                # If they are the same class, then save the image
                if imgClass == NAME_SUBFOLDER:
                    NAME_SUBFOLDER += "/"
                    NAME_IMG = trainImgsPaths[i]
                    frame = Image.open(NAME_IMG)
                    NAME_IMG = NAME_IMG.split("/")[-1]
                    frame.save(DB_PATH + TRAIN_SUBFOLDER +\
                                NAME_SUBFOLDER + NAME_IMG)
                else:
                    pass
        # Save test images
        # Read classes in testImgsClass
        for i in tqdm(range(len(testImgsClass))):
            imgClass = testImgsClass[i].split("/")[-1]
            for key in keys: 
                NAME_SUBFOLDER = key.split("/")[-1]
                # If they are the same class, then save the image
                if imgClass == NAME_SUBFOLDER:
                    NAME_SUBFOLDER += "/"
                    NAME_IMG = testImgsPaths[i]
                    frame = Image.open(NAME_IMG)
                    NAME_IMG = NAME_IMG.split("/")[-1]
                    frame.save(DB_PATH + VALIDATION_SUBFOLDER +\
                                NAME_SUBFOLDER + NAME_IMG)
                else:
                    pass
        ##########################################################

class preprocessImage:
    """
    Allows operations on single images.
    """
    def __init__(self):
        """
        Constructor of preprocessImage class
        """
        pass
        
    def divideIntoPatches(self,
                        imageWidth, 
                        imageHeight, 
                        slideWindowSize, 
                        strideSize, 
                        padding):
        """
        Divides the image into N patches depending on the stride size,
        the sliding window size and the type of padding.
        :param imageWidth: int that represents the width of the image
        :param imageHeight: int that represents the height of the image
        :param slideWindowSize: tuple (height, width) that represents the size 
                        of the sliding window
        :param strideSize: tuple (height, width) that represents the amount
                        of pixels to move on height and width direction 
        :param padding: string ("VALID", "SAME") that tells the type of 
                        padding
        : return: a list containing the number of patches that fill the
                given parameters, int containing the number of row patches,
                int containing the number of column patches
        """
        # Get sliding window sizes
        slideWindowHeight, slideWindowWidth  = slideWindowSize[0],\
                                                slideWindowSize[1]
        assert (slideWindowHeight < imageHeight) and (slideWindowWidth < imageWidth),\
                SLIDE_WINDOW_SIZE_TOO_BIG
        # Get strides sizes
        strideHeigth, strideWidth = strideSize[0], strideSize[1]
        assert (strideHeigth < imageHeight) and (strideWidth < imageWidth),\
                STRIDE_SIZE_TOO_BIG
        # Start padding operation
        if padding == 'VALID':
            startPixelsHeight = 0
            startPixelsWidth = 0
            patchesCoordinates = []
            numberPatchesHeight, numberPatchesWidth = getValidPadding(slideWindowHeight,
                                                                     strideHeigth,
                                                                     imageHeight,
                                                                     slideWindowWidth,
                                                                     strideWidth,
                                                                     imageWidth)
            print('numberPatchesHeight: ', numberPatchesHeight, 'numberPatchesWidth: ', numberPatchesWidth)
            for i in range(numberPatchesHeight):
                for j in range(numberPatchesWidth):
                    patchesCoordinates.append([startPixelsHeight,\
                                                    startPixelsWidth,\
                                                    slideWindowHeight,\
                                                    slideWindowWidth])
                    # Update width with strides 
                    startPixelsWidth += strideWidth
                    slideWindowWidth += strideWidth
                # Re-initialize the width parameters 
                startPixelsWidth = 0
                slideWindowWidth = slideWindowSize[1]
                # Update height with height stride size
                startPixelsHeight += strideHeigth 
                slideWindowHeight += strideHeigth
            return patchesCoordinates,\
                    numberPatchesHeight,\
                    numberPatchesWidth

        elif padding == 'SAME':
            startPixelsHeight = 0
            startPixelsWidth = 0
            patchesCoordinates = []
            # Modify image tensor 
            zeros_h, zeros_w = getSamePadding(slideWindowHeight,\
                                                strideHeigth,\
                                                imageHeight,\
                                                slideWindowWidth,\
                                                strideWidth,\
                                                imageWidth)
            imageWidth += zeros_w
            imageHeight += zeros_h
            # Valid padding strideHeigthould fit exactly
            numberPatchesHeight, numberPatchesWidth = getValidPadding(slideWindowHeight,\
                                                                    strideHeigth,\
                                                                    imageHeight,\
                                                                    slideWindowWidth,\
                                                                    strideWidth,\
                                                                    imageWidth)
            #######################TOFIX############################
            for i in range(numberPatchesHeight+1):
                for j in range(numberPatchesWidth+1):
            ########################################################
                    patchesCoordinates.append([startPixelsHeight,\
                                                startPixelsWidth,\
                                                slideWindowHeight,\
                                                slideWindowWidth])
                    # Update width with strides
                    startPixelsWidth += strideWidth
                    slideWindowWidth += strideWidth
                # Re-initialize width parameters
                startPixelsWidth = 0
                slideWindowWidth = slSize[1]
                # Update height with strides 
                startPixelsHeight += strideHeigth
                slideWindowHeight += strideHeigth
            #######################TOFIX############################
            return patchesCoordinates,\
                    numberPatchesHeight+1,\
                    numberPatchesWidth+1,\
                    zeros_h,\
                    zeros_w 
            ########################################################
        else:
            raise Exception("Type of padding not understood")
    
def getValidPadding(slideWindowHeight, 
                    strideHeigth, 
                    imageHeight, 
                    slideWindowWidth, 
                    strideWidth, 
                    imageWidth):
    """ 
    Given the dimensions of an image, the strides of the sliding window
    and the size of the sliding window. Find the number of patches that 
    fit in the image if the type of padding is VALID.
    :param slideWindowHeight: int that represents the height of the slide 
                                Window
    :param strideHeight: int that represents the height of the stride
    :param imageHeight: int that represents the height of the image
    :param slideWindowWidth: int that represents the width of the slide
                                window
    :param strideWidth: int that represents the width of the stride
    :param imageWidth: int that represents the width of the image
    : return: int that contains the number of patches in the height 
                dimension. Another int that contains the number of patches
                in the width dimension.
    """
    numberPatchesHeight_ = 0
    numberPatchesWidth_ = 0
    while(slideWindowHeight < imageHeight):
        slideWindowHeight += strideHeigth
        numberPatchesHeight_ += 1
    while(slideWindowWidth < imageWidth):
        slideWindowWidth += strideWidth
        numberPatchesWidth_ += 1
    return numberPatchesHeight_, numberPatchesWidth_

def getSamePadding(slideWindowHeight, 
                    strideHeigth, 
                    imageHeight, 
                    slideWindowWidth, 
                    strideWidth, 
                    imageWidth):
    """ 
    Given the dimensions of an image, the strides of the sliding window
    and the size of the sliding window. Find the number of zeros needed
    for the image so the sliding window fits as type of padding SAME. 
    Then find the number of patches that fit in the image. 
    :param slideWindowHeight: int that represents the height of the slide 
                                Window
    :param strideHeight: int that represents the height of the stride
    :param imageHeight: int that represents the height of the image
    :param slideWindowWidth: int that represents the width of the slide
                                window
    :param strideWidth: int that represents the width of the stride
    :param imageWidth: int that represents the width of the image
    : return: int zeros_h that represents the amount of zeros
                to add in the height dimension. int zeros_w 
                that represents the amount of zeros to add 
                in the width dimension. 
    """
    aux_slideWindowHeight = slideWindowHeight
    aux_slideWindowWidth = slideWindowWidth
    numberPatchesHeight_ = 0
    numberPatchesWidth_ = 0
    while(imageHeight > slideWindowHeight):
        slideWindowHeight += strideHeigth
        numberPatchesHeight_ += 1
    while(imageWidth > slideWindowWidth):
        slideWindowWidth += strideWidth
        numberPatchesWidth_ += 1
    # Pixels left that do not fit in the kernel 
    resid_h = imageHeight - (slideWindowHeight-strideHeigth)
    resid_w = imageWidth - (slideWindowWidth-strideWidth)
    # Amount of zeros to add to the image 
    zeros_h = aux_slideWindowHeight - resid_h
    zeros_w = aux_slideWindowWidth - resid_w
    #print(slideWindowHeight, imageHeight, resid_h, zeros_h)
    # Return amount of zeros
    return zeros_h, zeros_w

def lazy_SAMEpad(frame,
                zeros_h,
                zeros_w):
    """
    Given an image and the number of zeros to be added in height 
    and width dimensions, this function fills the image with the 
    required zeros.
    :param frame: opencv image of 3 dimensions
    :param zeros_h: int that represents the amount of zeros to be added
                    in the height dimension
    :param zeros_w: int that represents the amount of zeros to be added 
                    in the width dimension
    : return: a new opencv image with the added zeros
    """
    rows, cols, d = frame.shape
    # If height is even or odd
    if (zeros_h % 2 == 0):
        zeros_h = int(zeros_h/2)
        frame = r_[np.zeros((zeros_h, cols, 3)), frame,\
                    np.zeros((zeros_h, cols, 3))]
    else:
        zeros_h += 1
        zeros_h = int(zeros_h/2)
        frame = r_[np.zeros((zeros_h, cols, 3)), frame,\
                    np.zeros((zeros_h, cols, 3))]

    rows, cols, d = frame.shape
    # If width is even or odd 
    if (zeros_w % 2 == 0):
        zeros_w = int(zeros_w/2)
        # Container 
        container = np.zeros((rows,(zeros_w*2+cols),3), np.uint8)
        container[:,zeros_w:container.strideHeigthape[1]-zeros_w:,:] = frame
        frame = container #c_[np.zeros((rows, zeros_w)), frame, np.zeros((rows, zeros_w))]
    else:
        zeros_w += 1
        zeros_w = int(zeros_w/2)
        container = np.zeros((rows,(zeros_w*2+cols),3), np.uint8)
        container[:,zeros_w:container.strideHeigthape[1]-zeros_w:,:] = frame
        frame = container #c_[np.zeros((rows, zeros_w, 3)), frame, np.zeros((rows, zeros_w, 3))]
    return frame

def drawGrid(frame,
            patches):
    """
    Draws the given patches on top of the input image 
    :param frame: opencv input image 
    :param patches: a list containing the coordinates of the patches
                    calculated for the image
    : return: opencv image named frame that contains the same input
                image but with a grid of patches draw on top. 
    """
    # Iterate through patches
    for i in range(len(patches)):
        # Get patch
        patch = patches[i]
        # "Decode" patch 
        startHeight, startWidth, endHeight, endWidth = patch[0], patch[1],\
                                                        patch[2], patch[3]
        # Draw grids
        cv2.rectangle(frame, (startWidth, startHeight),\
                        (endWidth, endHeight), (0, 0, 255), 12)
        roi = np.zeros([patch[2]-patch[0], patch[3]-patch[1], 3],\
                        np.uint8)
        # Paint them red
        roi[:,:,:] = (0,0,255)
        cv2.addWeighted(frame[patch[0]:patch[2],patch[1]:patch[3],:],\
                        0.5, roi, 0.5, 0, roi)
        frame[patch[0]:patch[2],patch[1]:patch[3],:] = roi
    return frame
