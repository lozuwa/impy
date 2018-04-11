# Impy (Images in python)
<p>Impy is a library used for deep learning projects that make use of image datasets.</p>
<ul>
  <li><strong>Email: </strong>lozuwaucb@gmail.com</li>
  <li><strong>Bug reports: </strong>https://github.com/lozuwa/impy/issues</li>
</ul>
It provides:
<ul>
  <li>Data augmentation methods for images with bounding boxes (the bounding boxes are also affected by the transformation so you don't have to label again.)</li>
  <li>Fast image preprocessing methods useful in a deep learning context. E.g: if your image is too big you need to divide it into patches.</li>
</ul>

<p>For now the installation requires you to clone the repository. Follow the next steps:</p>

<ol>
  <li>Go to your package path. E.g: for anaconda go to ../Anaconda3/envs/YOUR_ENVIRONMENT/lib/pythonVERSION_OF_YOUR_PYTHON/site-packages/</li>
  <li>Clone the library git clone  https://github.com/lozuwa/impy.git</li>
  <li></li>
</ol>

# Tutorial
<p>This tutorial teaches you how to use the data augmentation methods available in the library. Suppose you have an image dataset that follows the VOC format. It means
your folders are structured in the following order.</p>
<ul>
  <li>
    database/
    <ul>
      <li>images/</li>
      <li>annotations/
        <ul>
          <li>xmls/</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>
<p>Where images contains all your .jpg files, annotations contains another folder called xmls and this folder contains the .xml files.</p>
<p>Following this structure you would write a script like the following:</p>

'''python
# General purpose imports
import os
import xml.etree.ElementTree as ET
# Import impy
from impy.DataAugmentation import *

# Create an instance of the class
da = DataAugmentation()

def process_bounding_box(img, annt):
  tree = ET.parse(annt)
  root = tree.getroot()
  if root.find("object"):
    # Find the size
    size = root.find("size")
    width, height, depth = [int(each) for each in get_size(size)]
    # Find the object
    objects = root.findall("object")
    # Get the bounding box
    bndboxes = get_bndbox(objects)
    # Iterate over bounding boxes
    for name, c
    

def main():
  # Read your images
  images = [os.path.join("images", each) for each in os.listdir("images/")]
  # Read your annotations
  annotations = [os.path.join("annotations", each) for each in os.listdir("annotations/xmls/")]
  # Iterate over both images
  for img, annt in zip(images, annotations):
    process_bounding_box(img, annt)

if __name__ == "__main__":
  main()

'''

