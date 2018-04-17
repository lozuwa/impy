"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: A class that allows to load a dataset and perform 
useful operations with it.
TODO:
	- Save xml and img file make sure the extension is correct. Maybe
		images with png are added to this class.
"""
import os
import json
import math
from tqdm import tqdm
try:
	from .VectorOperations import *
except:
	from VectorOperations import *
try:
	from .ImagePreprocessing import *
except:
	from ImagePreprocessing import *
try:
	from .BoundingBoxAugmenters import *
except:
	from BoundingBoxAugmenters import *
try:
	from .ImageAnnotation import *
except:
	from ImageAnnotation import *
try:
	from .Util import *
except:
	from Util import *

prep = ImagePreprocessing()
bndboxAugmenter = BoundingBoxAugmenters()

class ImageLocalizationDataset(object):

	def __init__(self, images = None, annotations = None, databaseName = None):
		super(ImageLocalizationDataset, self).__init__()
		# Assert images and annotations
		if (not os.path.isdir(images)):
			raise Exception("Path to images does not exist.")
		if (not os.path.isdir(annotations)):
			raise Exception("Path to annotations does not exist.")
		if (databaseName == None):
			databaseName = "Unspecified"
		# Class variables
		self.images = images
		self.annotations = annotations
		self.databaseName = databaseName

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
				ImageLocalizationDataset.save_img_and_xml(frame = frame,
																					bndboxes = bdxs,
																					names = annotations[i].propertyOtherAnnotationName,
																					database_name = self.databaseName,
																					data_augmentation_type = "Unspecified",
																					origin_information = image,
																					output_image_directory = self.images,
																					output_annotation_directory = self.annotations)

	def applyDataAugmentation(self,
														augmentations = None):
		"""
		Applies one or multiple data augmentation methods to the dataset.
		Args:
			augmenter: A string with a path to a json file.
		Returns:
			None
		"""
		# Assertions 
		if (augmentations == None):
			raise ValueError("Augmenter parameter cannot be empty.")
		else:
			if (not os.path.isfile(augmentations)):
				raise Exception("Path to json file ({}) does not exist."\
													.format(augmentations))
		# Load configuration data
		data = json.load(open(augmentations))
		augmentationTypes = [i for i in data.keys()]
		if (assertSupportedAugmentationTypes(augmentationTypes)):
			# Iterate over images
			imgs_path = [os.path.join(self.images, each) for each in os.listdir(self.images)]
			for img in tqdm(imgs_path):
				# Extract name
				file_name = os.path.split(img)[1].split(".jpg")[0]
				# Create xml and img name
				img_name = os.path.join(self.images, file_name + ".jpg")
				xml_name = os.path.join(self.annotations, file_name + ".xml")
				imgAnt = ImageAnnotation(path = xml_name)
				boundingBoxes = imgAnt.propertyBoundingBoxes
				names = imgAnt.propertyNames
				# Apply configured augmentations
				__applyAugmentation__(image = img_name,
															boundingBoxes = boundingBoxes,
															names = names,
															data = data)
		else:
			raise Exception("Augmentation type not supported.")	

	def __applyAugmentation__(self,
														image = None,
														boundingBoxes = None,
														names = None,
														data = None):
		"""
		Private method. Applies the supported data augmentation methods 
		to a data point.
		Args:
			image: A string that contains a path to an image.
			boundingBoxes:  A list of lists that contains bounding boxes' 
											coordinates.
			names: A list of strings that contains the names of the bounding boxes.
			data: A hashmap of hashmaps.
		Returns:
			None
		"""
		# Assertions
		if (image == None):
			raise ValueError("Image parameter cannot be empty.")
		if (boundingBoxes == None):
			raise ValueError("Bounding Boxes parameter cannot be empty.")
		if (data == None):
			raise ValueError("Data parameter cannot be empty.")
		# Local variables
		frame = cv2.imread(image)
		# Apply augmentations
		if ("bounding_box_augmenters" in augmentationTypes):
			for bounding_box_augmentation in data["bounding_box_augmenters"]:
				if (bounding_box_augmentation == "scale"):
					# Apply scaling
					parameters = data["bounding_box_augmenters"]["scale"]
					frame, bndboxes = bndboxAugmenter.scale(frame = frame,
																											boundingBoxes = boundingBoxes,
																											size = parameters["size"],
																											interpolationMethod = parameters["interpolationMethod"])
					# Save scaling
					ImageLocalizationDataset.save_img_and_xml(frame = frame,
																										bndboxes = bndboxes,
																										names = names,
																										database_name = self.databaseName,
																										data_augmentation_type = "bounding_box_scale",
																										origin_information = image,
																										output_image_directory = self.images,
																										output_annotation_directory = self.annotations)
				elif (bounding_box_augmentation == "crop"):
					# Apply scaling
					parameters = data["bounding_box_augmenters"]["crop"]
					frame, bndboxes = bndboxAugmenter.scale(frame = frame,
																									boundingBoxes = boundingBoxes,
																									size = parameters["size"],
																									interpolationMethod = parameters["interpolationMethod"])
					# Save scaling
					ImageLocalizationDataset.save_img_and_xml(frame = frame,
																										bndboxes = bndboxes,
																										names = names,
																										database_name = self.databaseName,
																										data_augmentation_type = "bounding_box_crop",
																										origin_information = image,
																										output_image_directory = self.images,
																										output_annotation_directory = self.annotations)
				elif (bounding_box_augmentation == "pad"):
					parameters = data["bounding_box_augmenters"]["pad"]
				elif (bounding_box_augmentation == "jitterBoxes"):
					parameters = data["bounding_box_augmenters"]["jitterBoxes"]
				elif (bounding_box_augmentation == "horizontalFlip"):
					parameters = data["bounding_box_augmenters"]["horizontalFlip"]
				elif (bounding_box_augmentation == "verticalFlip"):
					parameters = data["bounding_box_augmenters"]["verticalFlip"]
				elif (bounding_box_augmentation == "rotation"):
					parameters = data["bounding_box_augmenters"]["rotation"]
				elif (bounding_box_augmentation == "dropout"):
					parameters = data["bounding_box_augmenters"]["dropout"]
				else:
					pass
		if ("image_augmenters" in augmentationTypes):
			for bounding_box_augmentation in data["image_augmenters"]:
				if (bounding_box_augmentation == "scale"):
					parameters = data["image_augmenters"]["scale"]
				elif (bounding_box_augmentation == "translate"):
					parameters = data["image_augmenters"]["translate"]
				elif (bounding_box_augmentation == "jitterBoxes"):
					parameters = data["image_augmenters"]["jitterBoxes"]
				elif (bounding_box_augmentation == "horizontalFlip"):
					parameters = data["image_augmenters"]["horizontalFlip"]
				elif (bounding_box_augmentation == "verticalFlip"):
					parameters = data["image_augmenters"]["verticalFlip"]
				elif (bounding_box_augmentation == "rotation"):
					parameters = data["image_augmenters"]["rotation"]
				elif (bounding_box_augmentation == "invertColor"):
					parameters = data["image_augmenters"]["invertColor"]
				elif (bounding_box_augmentation == "histogramEqualization"):
					parameters = data["image_augmenters"]["histogramEqualization"]
				elif (bounding_box_augmentation == "changeBrightness"):
					parameters = data["image_augmenters"]["changeBrightness"]
				elif (bounding_box_augmentation == "sharpening"):
					parameters = data["image_augmenters"]["sharpening"]
				elif (bounding_box_augmentation == "addGaussianNoise"):
					parameters = data["image_augmenters"]["addGaussianNoise"]
				elif (bounding_box_augmentation == "gaussianBlur"):
					parameters = data["image_augmenters"]["gaussianBlur"]
				elif (bounding_box_augmentation == "shiftColors"):
					parameters = data["image_augmenters"]["shiftColors"]
				elif (bounding_box_augmentation == "fancyPCA"):
					parameters = data["image_augmenters"]["fancyPCA"]
				else:
					pass

	@staticmethod
	def assertSupportedAugmentationTypes(augmentationTypes = None):
		"""
		Assert json file contains supported features.
		Args:
			augmentationTypes: A list of strings with the augmentation types
												written in a json file.
		Returns:
			A boolean that tells if the information is supported.
		"""
		# Assertions
		if (augmentationTypes == None):
			raise ValueError("Augmentation types cannot be empty.")
		# Assert augmentation types are supported.
		for each in augmentationTypes:
			if (each in ["bounding_box_augmenters", "image_augmenters"]):
				pass
			else:
				return False
		return True
	
	@staticmethod
	def save_img_and_xml(frame = None,
												bndboxes = None,
												names = None,
												database_name = None,
												data_augmentation_type = None,
												origin_information = None,
												output_image_directory = None,
												output_annotation_directory = None):
		"""
		Saves an image and its annotation.
		Args:
			database_name: A string that contains the name of the database.
			frame_size: A numpy-tensorflow tensor that contains an image.
			bndboxes: A list of lists that contains the bounding boxes' coordinates.
			names: A list of strings that contains the labels of the bounding boxes.
			origin_information: A string that contains information about the file's 
													origin.
			data_augmentation_type: A string that contains the type of data augmentation.
			output_image_directory: A string that contains the path to save the image.
			output_annotation_directory: A string that contains the path to save the 
																	image's annotation.
		Returns:
			None
		"""
		# Assertions
		if (database_name == None):
			database_name = "Unspecified"
		if (frame == None):
			raise ValueError("Frame parameter cannot be empty.")
		if (bndboxes == None):
			raise ValueError("Bounding boxes parameter cannot be empty.")
		if (names == None):
			raise ValueError("Names parameter cannot be empty.")
		if (origin_information == None):
			origin_information = "Unspecified"
		if (data_augmentation_type == None):
			data_augmentation_type = "Unspecified"
		if (output_image_directory == None):
			raise ValueError("Output image directory directory parameter cannot be empty.")
		if (output_annotation_directory == None):
			raise ValueError("Output annotation directory directory parameter cannot be empty.")
		# Local variables
		# Generate a new name.
		new_name = Util.create_random_name(name = name, length = 4)
		img_name = new_name + ".jpg"
		xml_name = new_name + ".xml"
		# Save image
		img_save_path = os.path.join(output_image_directory, img_name)
		cv2.imwrite(img_save_path, frame)
		# Create and save annotation
		ImageLocalizationDataset.to_xml(filename = img_name,
																		path = img_save_path,
																		database_name = database_name,
																		frame_size = frame.shape,
																		data_augmentation_type = data_augmentation_type,
																		bounding_boxes = bndboxes,
																		names = names,
																		origin = origin_information,
																		output_directory = output_annotation_directory)

	@staticmethod
	def to_xml(filename = None,
							path = None,
							database_name = None,
							frame_size = None,
							data_augmentation_type = None,
							bounding_boxes = None,
							names = None,
							origin = None,
							output_directory = None):
		"""
		Creates an XML file that contains the annotation's information of an image.
		This file's structure is based on the VOC format.
		Args:
			filename: A string that contains the name of a file.
			path: A string that contains the path to an image.
			database_name: A string that contains the name of a database.
			frame_size: A tuple that contains information about the size of an image.
			data_augmentation_type
			coordinates: A list of lists that contains the bounding boxes annotations.
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
		for name, coordinate in zip(names, coordinates):
			object_ = ET.SubElement(annotation, "object")
			ET.SubElement(object_, "name").text = str(name)
			# ET.SubElement(object_, "pose").text = "Unspecified"
			# ET.SubElement(object_, "truncated").text = "0"
			ET.SubElement(object_, "difficult").text = "0"
			bndbox = ET.SubElement(object_, "bndbox")
			xmin, ymin, xmax, ymax = coordinate
			ET.SubElement(bndbox, "xmin").text = str(xmin)
			ET.SubElement(bndbox, "ymin").text = str(ymin)
			ET.SubElement(bndbox, "xmax").text = str(xmax)
			ET.SubElement(bndbox, "ymax").text = str(ymax)
		# Write file
		tree = ET.ElementTree(annotation)
		tree.write(os.path.join(output_directory, filename.split(".jpg")[0]+".xml"))

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