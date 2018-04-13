"""
package: Images2Dataset
class: DataAugmentation
Email: lozuwaucb@gmail.com
Author: Rodrigo Loza
Description: Common data augmentation operations 
for an image.
Log:
  Novemeber, 2017 -> Re-estructured class.
  December, 2017 -> Researched most used data augmentation techniques.
  March, 2018 -> Coded methods.
  April, 2018 -> Redesigned the methods to support multiple bounding
                 boxes (traditional data augmentation tools.)
  April, 2018 -> Redefined list of augmenters:

  Input to all methods: Given an image with its bounding boxes.
  ---------------
  Space dimension
  ---------------
  1. Scaling
    Resize the image to (h' x w') and maintain the bounding boxes' sizes.
  2. Random crop
    Crop the bounding boxes using random coordinates.
  3. Random pad (also translation)
    Include exterior pixels to bounding boxes.
  4. Flip horizontally
    Flip the bounding boxes horizontally.
  5. Flip vertically
    Flip the bounding boxes vertically.
  6. Rotation
    Randomly rotates the bounding boxes.
  7. Jitter boxes
    Draws random color boxes inside the bounding boxes.
  8. Dropout
    Sets pixels to zero with probability P.
"""
# Libraries
from interface import implements
import math
import random
import cv2
import numpy as np
# Other libraries
try:
  from .ImagePreprocessing import *
except:
  from ImagePreprocessing import *
# Interface
try:
  from .BoundingBoxAugmentationMethods import *
except:
  from BoundingBoxAugmentationMethods import *

class BoundingBoxAugmenters(implements(BoundingBoxAugmentationMethods)):
  """
  BoundingBoxAugmenters class. This class implements a set of data augmentation
  tools for bouding boxes.
  IMPORTANT
  - This class assumes input images are numpy tensors that follow the opencv
  color format BGR.
  """
  def __init__(self):
    super(BoundingBoxAugmenters, self).__init__()
    # Create an object of ImagePreprocessing
    self.prep = ImagePreprocessing()

  def scale(self,
            frame = None,
            boundingBoxes = None,
            resizeSize = None,
            interpolationMethod = None):
    """
    Scales an image with its bounding boxes to another size while maintaing the 
    coordinates of the bounding boxes.
    Args:
      frame: A tensor that contains an image.
      boundingBoxes: A list of lists that contains the coordinates of the bounding
                      boxes that are part of the image.
      resizeSize: A tuple that contains the resizing values.
      interpolationMethod: Set the type of interpolation method. 
                            (INTER_NEAREST -> 0,
                            INTER_LINEAR -> 1, 
                            INTER_CUBIC -> 2, 
                            INTER_LANCZOS4 -> 4)
    Returns:
      An image that has been scaled and a list of lists that contains the new 
      coordinates of the bounding boxes.
    """
    # Local variable assertions
    if (resizeSize == None):
      raise ValueError("resizeSize cannot be empty.")
    elif (type(resizeSize) != tuple):
      raise ValueError("resizeSize has to be a tuple (width, height)")
    else:
      if (len(resizeSize) != 2):
        raise ValueError("resizeSize must be a tuple of size 2 (width, height)")
      else:
        resizeWidth, resizeHeight = resizeSize
        if (resizeWidth == 0 or resizeHeight == 0):
          raise ValueError("Neither width nor height can be 0.")
    if (interpolationMethod == None):
      interpolationMethod = 2
    # Local variables
    height, width, depth = frame.shape
    reduY = height / resizeHeight
    reduX = width / resizeWidth
    # Scale image
    frame = cv2.resize(frame.copy(), resizeSize, interpolationMethod)
    # Fix bounding boxes
    for i in range(len(boundingBoxes)):
      # Decode bounding box
      ix, iy, x, y = boundingBoxes[i]
      # Update values with the resizing factor
      ix, iy, x, y = ix // reduX, iy // reduY, x // reduX, y // reduY
      # Check variables are not the same as the right and bottom boundaries
      x, y = BoundingBoxAugmenters.checkBoundaries(x, y, width, height)
      # Update list
      boundingBoxes[i] = [i for i in map(int, [ix, iy, x, y])]
    # Return values
    return frame, boundingBoxes

  def randomCrop(self,
                boundingBoxes = None,
                size = None):
    """
    Crop a list of bounding boxes.
    Args:
      boundingBoxes: A list of lists that contains the coordinates of bounding 
                    boxes.
      size: A tuple that contains the size of the crops to be performed.
    Returns:
      A list of lists with the updated coordinates of the bounding boxes after 
      being cropped.
    """
    # Local variables
    if (boundingBoxes == None):
      raise ValueError("Bounding boxes parameter cannot be empty.")
    if (len(size) == 3):
      height, width, depth = size
    elif (len(size) == 2):
      height, width = size
    else:
      raise Exception("Specify a size for the crops.")
    # Iterate
    for i in range(len(boundingBoxes)):
      # Decode bndbox
      ix, iy, x, y = boundingBoxes[i]
      # Compute width and height
      widthBndbox, heightBndbox = (x - ix), (y - iy)
      # Generate random number for the x axis
      rix = int(ix + (np.random.rand()*((x - width) - ix + 1)))
      # Generate random number for the y axis
      riy = int(iy + (np.random.rand()*((y - height) - iy + 1)))
      # Compute crop
      boundingBoxes[i] = [rix, riy, rix + width, riy + height]
    # Return values
    return boundingBoxes

  def randomPad(self,
                frameHeight = None,
                frameWidth = None,
                boundingBoxes = None,
                size = None):
    """
    Includes pixels from outside the bounding box as padding.
    Args:
      frameHeight: An int that contains the height of the frame.
      frameWidth: An int that contains the width of the frame.
      boundingBoxes: A list of lists that contains coordinates of bounding boxes.
      size: A tuple that contains the size of pixels to pad the image with.
    Returns:
      A list of lists that contains the coordinates of the bounding
      boxes padded with exterior pixels of the parent image.
    """
    # Assertions
    if (frameHeight == None):
      raise ValueError("Frame height cannot be empty.")
    if (frameWidth == None):
      raise ValueError("Frame width cannot be empty.")
    if (boundingBoxes == None):
      raise ValueError("Bounding boxes cannot be empty.")
    if (len(size) == 1):
      padWidth, padHeight = size[0], 0
    if (len(size) == 2):
      padWidth, padHeight = size[0], size[1]
    # Start padding
    for i in range(len(boundingBoxes)):
      # Decode bounding box
      ix, iy, x, y = boundingBoxes[i]
      # Determine how much space is there to pad on each side.
      padLeft = ix
      padRight = frameWidth - x
      padTop = iy
      padBottom = frameHeight - y
      if ((padLeft + padRight) >= padWidth):
        pass
      else:
        padWidth = padLeft + padRight
      if ((padTop + padBottom) >= padHeight):
        pass
      else:
        padHeight = padTop + padBottom 
      # Generate random numbers
      padx = int(np.random.rand()*padWidth)
      pady = int(np.random.rand()*padHeight)
      paddingLeft = padx // 2
      paddingRight = padx - paddingLeft
      paddingTop = pady // 2
      paddingBottom = pady - paddingTop
      # print("*", paddingLeft, paddingRight)
      # print("**", paddingTop, paddingBottom)
      # Modify coordinates
      if ((ix - paddingLeft) < 0):
        ix = 0
      else:
        ix -= paddingLeft
      if ((iy - paddingTop) < 0):
        iy = 0
      else:
        iy -= paddingTop
      if ((x + paddingRight) >= frameWidth):
        x = frameWidth
      else:
        x += paddingRight
      if ((y + paddingBottom) >= frameHeight):
        y = frameHeight
      else:
        y += paddingBottom
      # Update bounding box.
      boundingBoxes[i] = [ix, iy, x, y]
    # Return bouding boxes.
    return boundingBoxes

  def jitterBoxes(self, 
                  frame = None, 
                  boundingBoxes = None,
                  size = None, 
                  quantity = None,
                  color = None):
    """
    Draws random jitter boxes in the bounding boxes.
    Args:
      frame: A tensor that contains an image.
      boundingBoxes: A list of lists that contains the coordinates of the boudning
                    boxes that belong to the frame.
      size: A tuple that contains the size of the jitter boxes to draw.
      quantity: An int that tells how many jitter boxes to draw inside 
              each bounding box.
      color: A 3-sized tuple that contains some RGB color. If default it is black.
    Returns:
      A tensor that contains an image altered by jitter boxes.
    """
    # Assertions
    try:
      if (frame == None):
        raise Exception("Frame cannot be empty.")
    except:
      pass
    if (boundingBoxes == None):
      raise ValueError("Bounding boxes cannot be empty.")
    if (quantity == None):
      quantity = 3
    if (size == None):
      raise ValueError("Size cannot be empty.")
    if (color == None):
      color = (0, 0, 0)
    # Local variables
    if (len(frame.shape) == 2):
      height, width = frame.shape
    else:
      height, width, depth = frame.shape
    # Iterate over bounding boxes.
    for bndbox in boundingBoxes:
      # Decode bndbox.
      ix, iy, x, y = bndbox
      # Draw boxes.
      for i in range(quantity):
        rix = int(ix + (np.random.rand()*((x - size[0]) - ix + 1)))
        riy = int(iy + (np.random.rand()*((y - size[1]) - iy + 1)))
        # Draw jitter boxes on top of the image.
        frame = cv2.rectangle(frame, (rix, riy), (rix+size[0], riy+size[1]), \
                              color, -1)
    # Return frame
    return frame

  def horizontalFlip(self,
                    frame = None,
                    boundingBoxes = None):
    """
    Flip a bouding box by its horizontal axis.
    Args:
      frame: A tensor that contains an image with its bounding boxes.
      boundingBoxes: A list of lists that contains the coordinates of the bounding
                      boxes that belong to the tensor.
    Returns:
      A tensor whose bounding boxes have been flipped by its horizontal axis.
    """
    # Assertions
    try:
      if (frame == None):
        raise Exception("Frame parameter cannot be empty.")
    except:
      pass
    if (boundingBoxes == None):
      raise Exception("Bounding boxes parameter cannot be empty.")
    # Flip only the pixels inside the bounding boxes
    for bndbox in boundingBoxes:
      # Decode bounding box
      ix, iy, x, y = bndbox
      # Flip ROI
      roi = cv2.flip(frame[iy:y, ix:x, :], 1)
      frame[iy:y, ix:x, :] = roi
    return frame

  def verticalFlip(self,
                  frame = None,
                  boundingBoxes = None):
    """
    Flip a bouding box by its vertical axis.
    Args:
      frame: A tensor that contains a cropped bouding box from its frame.
      boundingBoxes: A list of lists that contains the coordinates of the bounding
                      boxes that belong to the tensor.
    Returns:
      A tensor whose bounding boxes have been flipped by its vertical axis.
    """
    # Assertions
    try:
      if (frame == None):
        raise Exception("Frame parameter cannot be empty.")
    except:
      pass
    if (boundingBoxes == None):
      raise Exception("Bounding boxes parameter cannot be empty.")
    # Flip only the pixels inside the bounding boxes
    for bndbox in boundingBoxes:
      # Decode bounding box
      ix, iy, x, y = bndbox
      # Flip ROI
      roi = cv2.flip(frame[iy:y, ix:x, :], 0)
      frame[iy:y, ix:x, :] = roi
    return frame

  def randomRotation(self, frame = None, bndbox = None, theta = None):
    """
    Rotate a frame clockwise by random degrees. Random degrees
    is a number that is between 20-360.
    Args:
      frame: A tensor that contains an image.
      bndbox: A tuple that contains the ix, iy, x, y coordinates 
              of the bounding box in the image.
      theta: An int that contains the amount of degrees to move.
              Default is random.
    Returns:
      A tensor that contains the rotated image and a tuple
      that contains the rotated coordinates of the bounding box.
    """
    # Assertions
    try:
      if frame == None:
        raise Exception("Frame cannot be emtpy.")
    except:
      pass
    if bndbox == None:
      raise Exception("Bnbdbox cannot be empty")
    if theta == None:
      theta = (random.random() * math.pi) + math.pi / 3
    # Local variables
    thetaDegrees = theta * 180 / math.pi
    rows, cols, depth = frame.shape
    # print("Degrees: ", thetaDegrees)
    # print("Rows, cols: ", rows//2, cols//2)
    # Decode the bouding box
    ix, iy, x, y = bndbox
    # print("Original: ", bndbox)
    # Fix the y coordinate since matrix transformations
    # assume 0,0 is at the left bottom corner.
    iy, y = rows-iy, rows-y
    # print(ix, iy, x, y)
    # Center the coordinates with respect to the 
    # center of the image.
    ix, iy, x, y = ix-(cols//2), iy-(rows//2), x-(cols//2), y-(rows//2)
    # print("Centered: ", ix, iy, x, y)
    # print(ix, iy, x, y)
    # Write down coordinates
    p0 = [ix, iy]
    p1 = [x, iy]
    p2 = [ix, y]
    p3 = [x, y]
    # Compute rotations on coordinates
    p0[0], p0[1] = DataAugmentation.rotation_equations(p0[0], p0[1], theta)
    p1[0], p1[1] = DataAugmentation.rotation_equations(p1[0], p1[1], theta)
    p2[0], p2[1] = DataAugmentation.rotation_equations(p2[0], p2[1], theta)
    p3[0], p3[1] = DataAugmentation.rotation_equations(p3[0], p3[1], theta)
    # Add centers to compensate
    p0[0], p0[1] = p0[0] + (cols//2), rows - (p0[1] + (rows//2))
    p1[0], p1[1] = p1[0] + (cols//2), rows - (p1[1] + (rows//2))
    p2[0], p2[1] = p2[0] + (cols//2), rows - (p2[1] + (rows//2))
    p3[0], p3[1] = p3[0] + (cols//2), rows - (p3[1] + (rows//2))
    # Rotate image
    M = cv2.getRotationMatrix2D((cols/2, rows/2), thetaDegrees, 1)
    frame = cv2.warpAffine(frame, M, (cols, rows))
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    ix, x = min(xs), max(xs)
    iy, y = min(ys), max(ys)
    # print(p0, p1, p2, p3)
    # Make sure ix, iy, x, y are valid
    if ix < 0:
      # If ix is smaller, then it was impossible to place 
      # the coordinate inside the image because of the angle. 
      # In this case, the safest option is to set ix to 0.
      print("WARNING: ix is negative.", ix)
      ix = 0
    if iy < 0:
      # If iy is smaller, then it was impossible to place 
      # the coordinate inside the image because of the angle. 
      # In this case, the safest option is to set iy to 0.
      print("WARNING: iy is negative.", iy)
      iy = 0
    if x >= cols:
      print("WARNING: x was the width of the frame.", x, cols)
      x = cols - 1
    if y >= rows:
      print("WARNING: y was the height of the frame.", y, rows)
      y = rows - 1
    # Return frame and coordinates
    return frame, [ix, iy, x, y]

  def dropout(self,
              frame = None,
              boundingBoxes = None,
              size = None,
              threshold = None,
              color = None):
    """
    Set pixels inside a bounding box to zero depending on probability p 
    extracted from a normal distribution with zero mean and one standard deviation.
    Args:
      frame: A tensor that contains an image.
      boundingBoxes: A list of lists that contains the coordinates of the bounding
                      boxes that belong to the frame.
      size: A tuple that contains the size of the regions that will be randomly
            set to zero according to a dropout scenario.
      threshold: A float that contains the probability threshold for the dropout
                  scenario.
    Returns:
      A tensor with the altered pixels.
    """
    # Assertions
    if (boundingBoxes == None):
      raise ValueError("Bounding boxes parameter cannot be empty.")
    if (size == None):
      raise ValueError("Size parameter cannot be empty.")
    if (threshold == None):
      threshold = 0.5
    else:
      if (threshold > 0.99):
        threshold = 0.99
    if (color == None):
      color = (0,0,0)
    # Iterate over bounding boxes
    for i in range(len(boundingBoxes)):
      # Decode bndbox
      ix, iy, x, y = boundingBoxes[i]
      # Preprocess image
      croppingCoordinates, _, \
                        __ = self.prep.divideIntoPatches(imageWidth = (x-ix),
                                                        imageHeight = (y-iy),
                                                        slideWindowSize = size,
                                                        strideSize = size,
                                                        padding = "VALID")
      for j in range(len(croppingCoordinates)):
        ixc, iyc, xc, yc = croppingCoordinates[j]
        rix, riy, rxc, ryc = ixc+ix, iyc+iy, ixc+ix+size[0], iyc+iy+size[1]
        prob = np.random.rand()
        if (prob > threshold):
          frame[riy:ryc, rix:rxc, :] = color
    return frame

  @staticmethod
  def rotation_equations(x, y, theta):
    """
    Apply a 2D rotation matrix to a 2D coordinate by theta degrees.
    Args:
      x: An int that represents the x dimension of the coordinate.
      y: An int that represents the y dimension of the coordinate.
    """
    x_result = int((x*math.cos(theta)) - (y*math.sin(theta)))
    y_result = int((x*math.sin(theta)) + (y*math.cos(theta)))
    return x_result, y_result

  @staticmethod
  def checkBoundaries(x = None, y = None, width = None, height = None):
    """
    Checks if the boundaries are in good shape.
    Args:
      x: An int that contains a coordinate.
      y: An int that contains a coordinate.
      width: An int that contains the x boundary of a frame.
      height: An int that contains the y boundary of a frame.
    """
    # End boundaries
    if (x == width):
      x -= 1
    if (y == height):
      y -= 1
    return x, y

# s = -3.79
# r = 0.758
# aux = []
# aux1 = []
# for i in range(10):
#   for each in l:
#     if ((each >= s) and (each <= (s + r))):
#       aux.append(each)
#   aux1.append(aux)
#   aux = []
#   s += r
