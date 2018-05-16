"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Implements common file, logic operations.
"""
import os
import datetime
import re
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET
# Local modules.
try:
  from .AssertDataTypes import *
except:
  from AssertDataTypes import *

class Util(object):
  def __init__(self):
    super(Util, self).__init__()

  @staticmethod
  def create_folder(folder_name = None):
    """
    Creates a folder.
    Args:
      folder_name: A string that contains the name of the folder to be created.
    Returns:
      None
    """
    # Assertions
    if (folder_name == None):
      raise ValueError("ERROR: Folder name parameter cannot be emtpy.")
    if (os.path.isdir(folder_name)):
      print("INFO: Folder already exists: {}".format(os.path.split(folder_name)[1]))
    else:
      try:
        os.mkdir(folder_name)
      except:
        raise Exception("ERROR: Folder {} could not be created.".format(folder_name))

  @staticmethod
  def create_random_name(name = None, length = None):
    """
    Generates a name based on the name parameter.
    Args:
      name: A string that contains the name of the image that is being labeled.
      length: An int that defines the length of the random string in the name
              to be created.
    Returns:
      A string that contains a new name with respect to the current time.
    """
    # Assertions
    if (name == None):
      raise ValueError("Name parameter cannot be empty.")
    if (length == None):
      raise ValueError("Length parameter cannot be empty.")
    # Local variables
    abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", 
      "m", "n", "o", "p", "q","r", "s", "t", "u", "v", "w", "x", "y", "z"]
    # Get current time.
    now = datetime.datetime.now()
    # Create random 4-length string.
    var_str = ""
    for i in range(4):
      random_index = int(np.random.rand()*(len(abc)-1))
      var_str += abc[random_index]
    # Append created variables.
    new_name = "{}_{}_{}_{}_{}_{}".format(name,
                                          now.year,
                                          now.month,
                                          now.hour,
                                          now.microsecond,
                                          var_str)
    # Return string
    return new_name

  @staticmethod
  def detect_file_extension(filename = None):
    """
    Detect the extension of a file.
    Args:
      filename: A string that contains the name of a file.
    Returns:
      A string that contains the extension of the file. It 
      returns None if there is no extension.
    """
    # Assertions
    if (filename == None):
      raise ValueError("Filename cannot be empty.")
    # Logic
    if (filename.endswith(".jpg")):
      return ".jpg"
    elif (filename.endswith(".png")):
      return ".png"
    else:
      return None

  @staticmethod
  def save_img(frame = None, img_name = None, output_image_directory = None):
    """
    Saves an image and its annotation.
    Args:
      frame: A numpy/tensorflow tensor that contains an image.
      img_name: A string with a name that contains an image extension.
      output_image_directory: A string that contains the path to save the image.
    Returns:
      None
    Raises:
      -Exception: In case the file was not written to disk.
    """
    # Assertions.
    if (assertNumpyType(data = frame) == False):
      raise ValueError("Frame has to be a numpy array.")
    if (img_name == None):
      raise ValueError("img_name cannot be emtpy.")
    extension = Util.detect_file_extension(filename = img_name)
    if (extension == None):
      raise Exception("Your image extension is not valid. " +\
                      "Only jpgs and pngs are allowed. {}".format(extension))
    # Local variables.
    img_save_path = os.path.join(output_image_directory, img_name)
    # Logic.
    cv2.imwrite(img_save_path, frame)
    # Assert file has been written to disk. 
    if (not os.path.isfile(img_save_path)):
      raise Exception("ERROR: Image was not saved. This happens " +\
                "sometimes when there are dozens of thousands of data " +\
                "points. Please run the script again and report this problem.")

  @staticmethod
  def save_annotation(filename = None, path = None, database_name = None, frame_size = None, data_augmentation_type = None, bounding_boxes = None, names = None, origin = None, output_directory = None):
    """
    Creates an XML file that contains the annotation's information of an image.
    This file's structure is based on the VOC format.
    Args:
      filename: A string that contains the name of a file. (Usually the name of the image).
      path: A string that contains the path to an image.
      database_name: A string that contains the name of a database.
      frame_size: A tuple that contains information about the size of an image.
      data_augmentation_type: A string that contains the type of augmentation that
                              is being used. Otherwise "Unspecified".
      bounding_boxes: A list of lists that contains the bounding boxes annotations.
      names: A list of strings that is parallel to bounding boxes. It depicts 
            the name associated with each bounding box.
      origin: A string that contains information about the origin of the file.
      output_directory: A string that contains the path to a directory to save 
                        the annotation.
    Returns:
      None
    """
    # Assertions
    if (filename == None):
      raise ValueError("Filename parameter cannot be empty.")
    if (path == None):
      raise ValueError("Path parameter cannot be empty.")
    if (database_name == None):
      raise ValueError("Database parameter cannot be empty.")
    if (frame_size == None):
      raise ValueError("Frame size parameter cannot be empty.")
    if (data_augmentation_type == None):
      raise ValueError("Data augmentation type parameter cannot be empty.")
    if (bounding_boxes == None):
      raise ValueError("Bounding boxes parameter cannot be empty.")
    if (names == None):
      raise ValueError("Names parameter cannot be empty.")
    if (origin == None):
      raise ValueError("Origin parameter cannot be empty.")
    if (output_directory == None):
      raise ValueError("Output directory parameter cannot be empty.")
    # XML VOC format
    annotation = ET.Element("annotation")
    # Image info
    ET.SubElement(annotation, "filename").text = str(filename)
    ET.SubElement(annotation, "origin").text = str(origin)
    ET.SubElement(annotation, "path").text = str(path)
    # Source
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = str(database_name)
    # Size
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "height").text = str(frame_size[0])
    ET.SubElement(size, "width").text = str(frame_size[1])
    if (len(frame_size) == 3):
      ET.SubElement(size, "depth").text = str(frame_size[2])
    # Data augmentation
    data_augmentation = ET.SubElement(annotation, "data_augmentation")
    ET.SubElement(data_augmentation, "type").text = str(data_augmentation_type)
    # Segmented
    ET.SubElement(annotation, "segmented").text = "0"
    # Objects
    for name, coordinate in zip(names, bounding_boxes):
      object_ = ET.SubElement(annotation, "object")
      ET.SubElement(object_, "name").text = str(name)
      ET.SubElement(object_, "pose").text = "Unspecified"
      ET.SubElement(object_, "truncated").text = "0"
      ET.SubElement(object_, "difficult").text = "0"
      bndbox = ET.SubElement(object_, "bndbox")
      xmin, ymin, xmax, ymax = coordinate
      ET.SubElement(bndbox, "xmin").text = str(xmin)
      ET.SubElement(bndbox, "ymin").text = str(ymin)
      ET.SubElement(bndbox, "xmax").text = str(xmax)
      ET.SubElement(bndbox, "ymax").text = str(ymax)
    # Write file
    tree = ET.ElementTree(annotation)
    extension = Util.detect_file_extension(filename)
    if (extension == None):
      raise Exception("Image's extension not supported {}".format(filename))
    tree.write(output_directory)
    # Assert file has been written to disk.
    if (not os.path.isfile(output_directory)):
      print(origin_information)
      print(img_name)
      print(xml_name)
      raise Exception("ERROR: Annotation was not saved. This happens " +\
                      "sometimes when there are dozens of thousands of data " +\
                      "points. Please run the script again and report this problem.")

  @staticmethod
  def save_lists_in_dataframe(columns = None, data = None, output_directory = None):
    """
    Save lists into a dataframe.
    Args:
      columns: A list of strings that contains the names of the columns 
              for the dataframe.
      data: A list of lists that contains data.
      output_directory: A string that contains the path to where save 
                        the dataframe.
    Returns:
      None
    """
    # Assertions
    if (columns == None):
      raise ValueError("ERROR: Paramater columns cannot be empty.")
    if (data == None):
      raise ValueError("ERROR: Paramater data cannot be empty.")
    if (output_directory == None):
      raise ValueError("ERROR: Paramater output_directory cannot be empty.")
    if (not os.path.isdir(output_directory)):
      raise Exception("ERROR: Path to {} does not exist.".format(output_directory))
    if (len(columns) != len(data)):
      raise Exception("ERROR: The len of the columns has to be the" +\
                      " same as data. Report this problem.")
    # Local import
    try:
      import pandas as pd
    except Exception as e:
      raise ImportError("ERROR: Pandas is not available, install it.")
    # Logic
    hashMap = {}
    for i in range(columns):
      hashMap[columns[i]] = data[i]
    df = pd.DataFrame(hashMap)
    df.to_excel(output_directory)

# import os
# import cv2
# import numpy as np

# images = os.listdir(".")
# hs = []
# ws = []
# for each in images:
#   frame = cv2.imread(each)
#   h = frame.shape[0]
#   w = frame.shape[1]
#   hs.append(h)
#   ws.append(w)

# hs = np.array(hs)
# ws = np.array(ws)

# print(hs.mean(), ws.mean())

# for each in images:
#   frame = cv2.imread(each)
#   cv2.imwrite(each, cv2.resize(frame, (int(ws.mean()), int(hs.mean()))))

