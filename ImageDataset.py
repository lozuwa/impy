"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: 
"""
import os
from tqdm import tqdm
import math
try:
	from .VectorOperations import *
except:
	from VectorOperations import *
try:
	from .ImagePreprocessing import *
except:
	from ImagePreprocessing import *
try:
	from .ImageAnnotation import *
except:
	from ImageAnnotation import *
try:
	from .Util import *
except:
	from Util import *

prep = ImagePreprocessing()

class ImageDataset(object):

	def __init__(self, images = None, annotations = None):
		super(ImageDataset, self).__init__()
		# Assert images and annotations
		if (not os.path.isdir(images)):
			raise Exception("Path to images does not exist.")
		if (not os.path.isdir(annotations)):
			raise Exception("Path to annotations does not exist.")
		# Class variables
		self.images = images
		self.annotations = annotations

	def reduceDatasetByRois(self,
													offset = None,
													output_directory = None):
		"""
		Example:
			Given an image and its bounding boxes, create ROIs of size offset
			that enclose the maximum possible amount of bounding boxes. 
				---------------------------------     	--------------------------------
				|                               |      |                               |
				|     ---                       |      |    Roi0------                 |
				|     | |                       |      |     |  |     |                |
				|     ---                       |      |     |---     |                |
				|                               |      |     |    --- |                |
				|            ---                |  ->  |     |    | | |                |
				|            | |                |      |     |    --- |                |
				|            ---                |      |     ------Roi0                |
				|                               |      |                               |
				|                               |      |                               |
				|                               |      |                               |
				|                  ---          |      |                 Roi1----      |
				|                  | |          |      |                 |      |      |
				|                  ---          |      |                 |      |      |
				|                               |      |                 |  --- |      |
				|                               |      |                 |  | | |      |
				|                               |      |                 |  --- |      |
				|                               |      |                 ----Roi1      |
				---------------------------------      ---------------------------------
		Then, the rois are saved with their respective annotations.
		"""
		# Create folders
		os.system("rm -r {}/reducedDataset".format(output_directory))
		Util.create_folder(os.path.join(output_directory, "reducedDataset"))
		Util.create_folder(os.path.join(output_directory, "reducedDataset", "images"))
		Util.create_folder(os.path.join(output_directory, "reducedDataset", "annotations"))
		Util.create_folder(os.path.join(output_directory, "reducedDataset", "annotations", "xmls"))
		# Get images and annotations full paths
		imgs_path = [os.path.join(self.images, each) for each in os.listdir(self.images)]
		for img in tqdm(imgs_path):
			#print(img)
			# Extract name
			file_name = os.path.split(img)[1].split(".jpg")[0]
			# Create xml and img name
			img_name = os.path.join(self.images, file_name + ".jpg")
			xml_name = os.path.join(self.annotations, file_name + ".xml")
			self.reduceImageDataPointByRoi(image = img_name, 
																			annotation = xml_name,
																			offset = offset,
																			output_directory = output_directory)

	def reduceImageDataPointByRoi(self,
																image = None,
																annotation = None,
																offset = None,
																output_directory = None):
		"""
		Args:
			image: A string that contains the path to an image.
			annotation: A string that contains the path to an annotation.
		"""	
		# Process image annotation
		annotation = ImageAnnotation(path = annotation)
		width, height, depth = annotation.propertySize
		names = annotation.propertyNames
		objects = annotation.propertyObjects
		boundingBoxes = annotation.propertyBoundingBoxes

		# Create a list of classes with the annotations.
		annotations = []
		index = 0
		for boundingBox, name in zip(boundingBoxes, names):
			# Compute the module
			ix, iy, x, y = boundingBox
			module = VectorOperations.compute_module(vector = [ix, iy])
			annotations.append(Annotation(index = index, name = name, \
																		bndbox = boundingBox, module = module,
																		inUse = False))
			index += 1

		# Sort the list of Annotations by its module from lowest to highest.
		for i in range(len(annotations)):
			for j in range(len(annotations)-1):
				module0 = annotations[i].propertyModule
				module1 = annotations[j].propertyModule
				if (module1 > module0):
					# Update Annotation's index
					annotations[i].propertyIndex = j
					annotations[j].propertyIndex = i
					# Swap Annotation
					aux = annotations[i]
					annotations[i] = annotations[j]
					annotations[j] = aux

		# Debug
		# for each in annotations:
		# 	print(each.propertyName, each.propertyModule, each.propertyIndex)
		# print("\n")

		# Find annotations that are close to each other.
		for i in range(len(annotations)):
			if (annotations[i].propertyInUse == False):
				# print(annotations[i].propertyName)
				ix0, iy0, x0, y0 = annotations[i].propertyBndbox
				annotations[i].includeOtherAnnotation([ix0, iy0, x0, y0])
				annotations[i].includeOtherAnnotationName(annotations[i].propertyName)
				annotations[i].propertyInUse = True
				for j in range(len(annotations)):
					ix1, iy1, x1, y1 = annotations[j].propertyBndbox
					if ((ix0 < ix1) and (iy0 < iy1)):
						# print(annotations[j].propertyName)
						distance = VectorOperations.euclidean_distance(v0 = [ix0, iy0], v1 = [x1, y1])
						if (distance < (offset-20)):
							annotations[i].includeOtherAnnotation([ix1, iy1, x1, y1])
							annotations[i].includeOtherAnnotationName(annotations[j].propertyName)
							annotations[j].propertyInUse = True

		# Debug
		# for each in annotations:
		# 	print(each.propertyIndex, each.propertyOtherAnnotation, "\n")
		# print("\n")

		# Save image croppings
		for i in range(len(annotations)):
			if (len(annotations[i].propertyOtherAnnotation) == 0):
				continue
			else:
				# Adjust image
				RoiXMin, RoiYMin, RoiXMax,\
				RoiYMax, bdxs = prep.adjustImage(frameHeight = height,
																				frameWidth = width,
																				boundingBoxes = annotations[i].propertyOtherAnnotation,
																				offset = offset)
				# Read image
				frame = cv2.imread(image)
				# Save image
				self.save_img_and_xml(name = "SPID",
												frame = frame[RoiYMin:RoiYMax,RoiXMin:RoiXMax,:],
												bndboxes = bdxs,
												names = annotations[i].propertyOtherAnnotationName,
												data_augmentation_type = "normal",
												origin = image,
												output_directory = output_directory)

	def save_img_and_xml(self,
												name, 
												frame, 
												bndboxes, 
												names, 
												data_augmentation_type, 
												origin, 
												output_directory):
		# Local variables
		new_name = Util.create_random_name(name = name,
																			length = 4)
		img_name = new_name + ".jpg"
		xml_name = new_name + ".xml"
		# print(new_name)
		height, width, depth = frame.shape
		# Save image
		img_save_path = os.path.join(output_directory, "reducedDataset", "images", img_name)
		cv2.imwrite(img_save_path, frame)
		# Save xml
		self.to_xml(folder = "images",
						filename = img_name,
						path = "Not specified",
						database = "SPID",
						width = width,
						height = height,
						depth = 3,
						data_augmentation_type = data_augmentation_type,
						coordinates = bndboxes,
						names = names,
						origin = origin,
						output_directory = output_directory)

	def to_xml(self,
							folder,
							filename,
							path,
							database,
							width,
							height,
							depth,
							data_augmentation_type,
							coordinates,
							names,
							origin,
							output_directory):
		"""
		Converts the input values into a XML file with a PASCAL VOC annotation.
		Args:
			folder: A string that contains the name of the folder where the
							image is stored.
			filename: A string that contains the name of the file.
			path: A string that contains the path to the image.
			database: A string that contains the name of the database.
			width: An int that contains image"s width.
			height: An int that contains image"s height.
			depth: An int that contains image"s depth.
			coordinates: A list of lists that contains the bounding boxes annotations.
		"""
		# XML VOC format
		annotation = ET.Element("annotation")
		# Image info
		ET.SubElement(annotation, "folder", verified = "yes").text = str(folder)
		ET.SubElement(annotation, "filename").text = str(filename)
		ET.SubElement(annotation, "origin").text = str(origin)
		ET.SubElement(annotation, "path").text = str(path)
		# Source
		source = ET.SubElement(annotation, "source")
		ET.SubElement(source, "database").text = str(database)
		# Size
		size = ET.SubElement(annotation, "size")
		ET.SubElement(size, "width").text = str(width)
		ET.SubElement(size, "height").text = str(height)
		ET.SubElement(size, "depth").text = str(depth)
		# Data augmentation
		data_augmentation = ET.SubElement(annotation, "data_augmentation")
		ET.SubElement(data_augmentation, "type").text = str(data_augmentation_type)
		# Segmented
		ET.SubElement(annotation, "segmented").text = "0"
		# Objects
		for name, coordinate in zip(names, coordinates):
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
		tree.write(os.path.join(output_directory, "reducedDataset", "annotations", "xmls",\
														filename.split(".jpg")[0]+".xml"))

class Annotation(object):
	def __init__(self, index, name, bndbox, module, inUse):
		super(Annotation, self).__init__()
		self.index = index
		self.name = name
		self.bndbox = bndbox
		self.module = module
		self.inUse = inUse
		self.otherAnnotations = []
		self.otherAnnotationsName = []

	@property
	def propertyModule(self):
		return self.module

	@property
	def propertyIndex(self):
		return self.index

	@propertyIndex.setter
	def propertyIndex(self, index):
		self.index = index

	@property
	def propertyName(self):
		return self.name

	@property
	def propertyBndbox(self):
		return self.bndbox

	@property
	def propertyModule(self):
		return self.module

	@propertyModule.setter
	def propertyModule(self, module):
		self.module = module

	@property
	def propertyInUse(self):
		return self.inUse

	@propertyInUse.setter
	def propertyInUse(self, inUse):
		self.inUse = inUse

	@property
	def propertyOtherAnnotation(self):
		return self.otherAnnotations

	def includeOtherAnnotation(self, annt):
		self.otherAnnotations.append(annt)

	@property
	def propertyOtherAnnotationName(self):
		return self.otherAnnotationsName

	def includeOtherAnnotationName(self, name):
		self.otherAnnotationsName.append(name)