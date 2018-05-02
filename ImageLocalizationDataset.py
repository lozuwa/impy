"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: A class that allows to load a dataset and perform 
useful operations with it.
"""
import os
import json
import math
import numpy as np
from tqdm import tqdm
try:
	from .ImagePreprocessing import *
except:
	from ImagePreprocessing import *

try:
	from .BoundingBoxAugmenters import *
except:
	from BoundingBoxAugmenters import *

try:
	from .ColorAugmenters import *
except:
	from ColorAugmenters import * 

try:
	from .ImageAnnotation import *
except:
	from ImageAnnotation import *

try:
	from .VectorOperations import *
except:
	from VectorOperations import *

try:
	from .Util import *
except:
	from Util import *

try:
	from .AssertDataTypes import *
except:
	from AssertDataTypes import *

try:
	from .AssertJsonConfiguration import *
except:
	from AssertJsonConfiguration import *

prep = ImagePreprocessing()
bndboxAugmenter = BoundingBoxAugmenters()
colorAugmenter = ColorAugmenters()
dataAssertion = AssertDataTypes()

class ImageLocalizationDataset(object):

	def __init__(self, imagesDirectory = None, annotationsDirectory = None, databaseName = None):
		"""
		A high level data structure used for image localization datasets.
		Args:
			imagesDirectory = None,
			annotationsDirectory = None,
			databaseName = None
		Returns:
			None
		"""
		super(ImageLocalizationDataset, self).__init__()
		# Assert images and annotations
		if (not os.path.isdir(imagesDirectory)):
			raise Exception("Path to images does not exist.")
		if (not os.path.isdir(annotationsDirectory)):
			raise Exception("Path to annotations does not exist.")
		if (databaseName == None):
			databaseName = "Unspecified"
		# Class variables
		self.imagesDirectory = imagesDirectory
		self.annotationsDirectory = annotationsDirectory
		self.databaseName = databaseName

	# Cleaning
	def dataConsistency(self):
		"""
		Checks whether data is consistent. It starts analizing if there is the same amount of 
		of images and annotations. Then it sees if the annotations and images are consistent 
		with each other.
		Args:
			None
		Returns:
			None
		"""
		# Local variables.
		images = []
		annotations = []
		# Preprocess images.
		for image in tqdm(os.listdir(self.imagesDirectory)):
			# Extract name.
			extension = Util.detect_file_extension(filename = image)
			if (extension == None):
				raise Exception("ERROR: Your image extension is not valid: {}".format(extension) +\
												 " Only jpgs and pngs are allowed.")
			images.append(image.split(extension)[0])
		# Preprocess annotations.
		for annotation in tqdm(os.listdir(self.annotationsDirectory)):
			if (not annotation.endswith(".xml")):
				raise Exception("ERROR: Only xml annotations are allowed: {}".format(annotation))
			annotations.append(annotation.split(".xml")[0])
		# Convert lists to sets.
		imagesSet = set(images)
		annotationsSet = set(annotations)
		# Check name consistency.
		imgToAnnt = imagesSet.difference(annotationsSet)
		anntToImg = annotationsSet.difference(imagesSet)
		# Check size consistency.
		if (len(imagesSet) != len(annotationsSet)):
			print("Images to annotations: ", imgToAnnt)
			print("Annotations to images: ", anntToImg)
			raise Exception("ERROR: The amount of images({}) and annotations({}) is not equal."\
											.format(len(imagesSet), len(annotationsSet)))
		if (len(imgToAnnt) != 0):
			raise Exception("ERROR: There are more images than annotations: {}".format(imgToAnnt))
		if (len(anntToImg) != 0):
			raise Exception("ERROR: There are more annotations than images: {}".format(anntToImg))

	def findEmptyOrWrongAnnotations(self, removeEmpty = None):
		"""
		Find empty or irregular annotations in the annotation files. An empty 
		annotation is an annotation that includes no objects. And a irregular 
		annotation is an annotation that has a bounding box with coordinates that
		are off the image's boundaries.
		Args:
			removeEmpty: A boolean that if True removes the annotation and image that are empty.
		Returns:
			None
		"""
		# Assertions
		if (removeEmpty == None):
			removeEmpty = False
		# Local variables
		emptyAnnotations = []
		files = os.listdir(self.imagesDirectory)
		# Logic
		for file in tqdm(files):
			extension = Util.detect_file_extension(filename = file)
			if (extension == None):
				raise Exception("ERROR: Your image extension is not valid: {}".format(extension) +\
												 " Only jpgs and pngs are allowed.")
			# Extract name
			filename = os.path.split(file)[1].split(extension)[0]
			# Create xml and img name
			imgFullPath = os.path.join(self.imagesDirectory, filename + extension)
			xmlFullPath = os.path.join(self.annotationsDirectory, filename + ".xml")
			# Create an object of ImageAnnotation.
			annt = ImageAnnotation(path = xmlFullPath)
			# Check if it is empty.
			if (len(annt.propertyBoundingBoxes) == 0):
				emptyAnnotations.append(file)
				print("WARNING: Annotation {} does not have any annotations.".format(xmlFullPath))
				# Check if we need to remove this annotation.
				if (removeEmpty == True):
					os.remove(imgFullPath)
					os.remove(xmlFullPath)
			# Check if it is irregular
			height, width, depth = annt.propertySize
			for each in annt.propertyBoundingBoxes:
				ix, iy, x, y = each
				if (ix < 0):
					raise ValueError("ERROR: Negative coordinate found in {}".format(file))
				if (iy < 0):
					raise ValueError("ERROR: Negative coordinate found in {}".format(file))
				if (x > width):
					raise ValueError("ERROR: Coordinate {} bigger than width {} found in {}"\
													.format(x, width, file))
				if (y > height):
					raise ValueError("ERROR: Coordinate {} bigger than height {} found in {}"\
														.format(y, height, file))
		# Return empty annotations
		return emptyAnnotations

	# Stats
	def computeBoundingBoxStats(self, saveDataFrame = None, outputDirDataFrame = None):
		"""
		Compute basic stats for the bounding boxes of the dataset.
		"""
		# Assertions
		if (saveDataFrame == None):
			saveDataFrame = False
		else:
			if (outputDirDataFrame == None):
				raise ValueError("Parameter directory dataframe cannot be empty.")
		# Local variables
		namesFrequency = {}
		files = os.listdir(self.imagesDirectory)
		columns = ["path", "name", "width", "height", "xmin", "ymin", "xmax", "ymax"]
		paths = []
		names = []
		widths = []
		heights = []
		boundingBoxesLists = []
		# Logic
		for file in tqdm(files):
			extension = Util.detect_file_extension(filename = file)
			if (extension == None):
				raise Exception("ERROR: Your image extension is not valid: {}".format(extension) +\
												 " Only jpgs and pngs are allowed.")
			# Extract name.
			filename = os.path.split(file)[1].split(extension)[0]
			# Create xml and img name.
			imgFullPath = os.path.join(self.imagesDirectory, filename + extension)
			xmlFullPath = os.path.join(self.annotationsDirectory, filename + ".xml")
			# Create an object of ImageAnnotation.
			annt = ImageAnnotation(path = xmlFullPath)
			# Check if it is empty.
			boundingBoxes = annt.propertyBoundingBoxes
			names = annt.propertyNames
			height, width, depth = annt.propertySize
			for i in range(len(names)):
				if (not (names[i] in namesFrequency)):
					namesFrequency[names[i]] = 0
				else:
					namesFrequency[names[i]] += 1
				paths.append(file)
				names.append(names[i])
				widths.append(width)
				heights.append(height)
				boundingBoxesLists.append(boundingBoxes[i])
		# Print stats
		print("Total number of bounding boxes: {}"\
						.format(sum([i for i in namesFrequency.values()])))
		print("Unique classes: {}".format(namesFrequency))
		# Save data?
		if (saveDataFrame):
			ImageLocalizationDataset.save_lists_in_dataframe(columns = columns,
									data = [paths, names, widths, heights, boundingBoxesLists],
									output_directory = outputDirDataFrame)

	# Preprocessing
	def reduceDatasetByRois(self, offset = None, outputImageDirectory = None, outputAnnotationDirectory = None):
		"""
		Reduce that images of a dataset by grouping its bounding box annotations and
		creating smaller images that contain them.
		Args:
			offset: An int that contains the amount of pixels in which annotations 
							can be grouped.
			outputImageDirectory: A string that contains the path to the directory
														where the images will be stored.  
			outputAnnotationDirectory: A string that contains the path to the directory
																where the annotations will be stored. 
		Returns:
			None
		"""
		# Assertions
		if (offset == None):
			raise ValueError("ERROR: Offset parameter cannot be empty.")
		if (outputImageDirectory == None):
			outputImageDirectory = os.getcwd()
			Util.create_folder(os.path.join(outputImageDirectory, "images"))
			outputImageDirectory = os.path.join(os.getcwd(), "images")
		if (not (os.path.isdir(outputImageDirectory))):
			raise Exception("ERROR: Path to output directory does not exist. {}"\
											.format(outputImageDirectory))
		if (outputAnnotationDirectory == None):
			outputAnnotationDirectory = os.getcwd()
			Util.create_folder(os.path.join(outputAnnotationDirectory, "annotations"))
			Util.create_folder(os.path.join(outputAnnotationDirectory, "annotations", "xmls"))
			outputAnnotationDirectory = os.path.join(os.getcwd(), "annotations", "xmls")
		if (not (os.path.isdir(outputAnnotationDirectory))):
			raise Exception("ERROR: Path to output annotation directory does not exist. {}"\
											.format(outputAnnotationDirectory))
		# Get images and annotations full paths
		imagesPath = [os.path.join(self.imagesDirectory, each) for each in \
									os.listdir(self.imagesDirectory)]
		for img in tqdm(imagesPath):
			#print(img)
			# Get extension
			extension = Util.detect_file_extension(filename = img)
			if (extension == None):
				raise Exception("ERROR: Your image extension is not valid." +\
												 "Only jpgs and pngs are allowed.")
			# Extract name
			filename = os.path.split(img)[1].split(extension)[0]
			# Create xml and img name
			imgFullPath = os.path.join(self.imagesDirectory, filename + extension)
			xmlFullPath = os.path.join(self.annotationsDirectory, filename + ".xml")
			self.reduceImageDataPointByRoi(imagePath = imgFullPath, 
																			annotationPath = xmlFullPath,
																			offset = offset,
																			outputImageDirectory = outputImageDirectory, 
																			outputAnnotationDirectory = outputAnnotationDirectory)

	def reduceImageDataPointByRoi(self, imagePath = None, annotationPath = None, offset = None, outputImageDirectory = None, outputAnnotationDirectory = None):
		"""
		Group an image's bounding boxes into Rois and create smaller images.
		Args:
			imagePath: A string that contains the path to an image.
			annotationPath: A string that contains the path to an annotation.
			offset: An int that contains the offset.
			outputImageDirectory: A string that contains the path where the images
														will be stored.
			outputAnnotationDirectory: A string that contains the path where the annotations
																will be stored.
		Returns:
			None
		Example:
			Given an image and its bounding boxes, create ROIs of size offset
			that enclose the maximum possible amount of bounding boxes. 
				---------------------------------       --------------------------------
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
		# Assertions
		if (imagePath == None):
			raise ValueError("ERROR: Path to imagePath parameter cannot be empty.")
		if (annotationPath == None):
			raise ValueError("ERROR: Path to annotation parameter cannot be empty.")
		if (not os.path.isfile(imagePath)):
			raise ValueError("ERROR: Path to image does not exist {}.".format(imagePath))
		if (not os.path.isfile(annotationPath)):
			raise ValueError("ERROR: Path to annotation does not exist {}.".format(annotationPath))
		if (offset == None):
			raise ValueError("ERROR: Offset parameter cannot be empty.")
		if (not (os.path.isdir(outputImageDirectory))):
			raise ValueError("ERROR: Output image directory does not exist.")
		if (not (os.path.isdir(outputAnnotationDirectory))):
			raise ValueError("ERROR: Output annotation directory does not exist.")
		# Load image annotation.
		annotation = ImageAnnotation(path = annotationPath)
		height, width, depth = annotation.propertySize
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
			annotations.append(Annotation(name = name, bndbox = boundingBox, \
																		module = module, corePoint = True))
			index += 1

		# Sort the list of Annotations by its module from lowest to highest.
		for i in range(len(annotations)):
			for j in range(len(annotations)-1):
				module0 = annotations[j].propertyModule
				module1 = annotations[j+1].propertyModule
				if (module0 >= module1):
					# Swap Annotation
					aux = annotations[j+1]
					annotations[j+1] = annotations[j]
					annotations[j] = aux

		# Debug
		# for each in annotations:
		#   print(each.propertyName, each.propertyModule)
		# print("\n")

		# Work on the points.
		for i in range(len(annotations)):
			# Ignore non-core points.
			if (annotations[i].propertyCorePoint == False):
				pass
			else:
				# Center the core point in an allowed image space.
				RoiXMin, RoiYMin, \
				RoiXMax, RoiYMax = prep.adjustImage(frameHeight = height,
																frameWidth = width,
																boundingBoxes = [annotations[i].propertyBndbox],
																offset = offset)
				# Find the annotations that can be included in the allowed image space.
				for j in range(len(annotations)):
					# Get bounding box.
					ix, iy, x, y = annotations[j].propertyBndbox
					# Check current bounding box is inside the allowed space.
					if ((ix >= RoiXMin) and (x <= RoiXMax)) and \
							((iy >= RoiYMin) and (y <= RoiYMax)):
							# Disable point from being a core point. Check it is not the 
							# current point of reference.
							if (not (annotations[i].propertyBndbox == annotations[j].propertyBndbox)):
								annotations[j].propertyCorePoint = False
				# Include the corresponding bounding boxes in the region of interest.
				newBoundingBoxes, \
				newNames = prep.includeBoundingBoxes(edges = [RoiXMin, RoiYMin, RoiXMax, RoiYMax],
																						boundingBoxes = boundingBoxes,
																						names = names)
				if (len(newBoundingBoxes) == 0):
					print(boundingBoxes)
					print(RoiXMin, RoiYMin, RoiXMax, RoiYMax)
					raise Exception("ERROR: No bounding boxes: {}. Please report this problem.".format(imagePath))
				# Read image
				frame = cv2.imread(imagePath)
				# Save image
				ImageLocalizationDataset.save_img_and_xml(frame = frame[RoiYMin:RoiYMax,\
																															RoiXMin:RoiXMax, :],
											bndboxes = newBoundingBoxes,
											names = newNames,
											database_name = self.databaseName,
											data_augmentation_type = "Unspecified",
											origin_information = imagePath,
											output_image_directory = outputImageDirectory,
											output_annotation_directory = outputAnnotationDirectory)

	def applyDataAugmentation(self, configurationFile = None, outputImageDirectory = None, outputAnnotationDirectory = None, threshold = None):
		"""
		Applies one or multiple data augmentation methods to the dataset.
		Args:
			configurationFile: A string with a path to a json file that contains the 
								configuration of the data augmentation methods.
			outputImageDirectory: A string that contains the path to the directory where
														images will be saved.
			outputAnnotationDirectory: A string that contains the path the directory where
																annotations will be saved.
			threshold: A float that contains a number between 0 and 1.
		Returns:
			None
		"""
		# Assertions 
		if (configurationFile == None):
			raise ValueError("ERROR: Augmenter parameter cannot be empty.")
		else:
			if (not os.path.isfile(configurationFile)):
				raise Exception("ERROR: Path to json file ({}) does not exist."\
													.format(configurationFile))
		jsonConf = AssertJsonConfiguration(file = configurationFile)
		typeAugmentation = jsonConf.runAllAssertions()
		if (outputImageDirectory == None):
			outputImageDirectory = os.getcwd()
			Util.create_folder(os.path.join(outputImageDirectory, "images"))
			outputImageDirectory = os.path.join(os.getcwd(), "images")
		if (not (os.path.isdir(outputImageDirectory))):
			raise Exception("ERROR: Path to output directory does not exist. {}"\
											.format(outputImageDirectory))
		if (outputAnnotationDirectory == None):
			outputAnnotationDirectory = os.getcwd()
			Util.create_folder(os.path.join(outputAnnotationDirectory, "annotations"))
			Util.create_folder(os.path.join(outputAnnotationDirectory, "annotations", "xmls"))
			outputAnnotationDirectory = os.path.join(os.getcwd(), "annotations", "xmls")
		if (not (os.path.isdir(outputAnnotationDirectory))):
			raise Exception("ERROR: Path to output annotation directory does not exist. {}"\
											.format(outputAnnotationDirectory))
		if (threshold == None):
			threshold = 0.5
		if (type(threshold) != float):
			raise TyperError("ERROR: threshold parameter must be of type float.")
		if ((threshold > 1) or (threshold < 0)):
			raise ValueError("ERROR: threshold paramater should be a number between" +\
												" 0-1.")
		# Load configuration data.
		f = open(configurationFile)
		data = json.load(f)
		f.close()
		# Iterate over the images.
		for img in tqdm(os.listdir(self.imagesDirectory)):
			# Get the extension
			extension = Util.detect_file_extension(filename = img)
			if (extension == None):
				raise Exception("ERROR: Your image extension is not valid." +\
												 "Only jpgs and pngs are allowed.")
			# Extract name.
			filename = os.path.split(img)[1].split(extension)[0]
			# Create xml and img name.
			imgFullPath = os.path.join(self.imagesDirectory, filename + extension)
			xmlFullPath = os.path.join(self.annotationsDirectory, filename + ".xml")
			imgAnt = ImageAnnotation(path = xmlFullPath)
			boundingBoxes = imgAnt.propertyBoundingBoxes
			names = imgAnt.propertyNames
			# Apply augmentation.
			if (typeAugmentation == 0):
				for i in data["bounding_box_augmenters"]:
					if (i == "Sequential"):
						# Prepare data for sequence
						frame = cv2.imread(imgFullPath)
						bndboxes = boundingBoxes
						# Read elements of vector
						assert type(data["bounding_box_augmenters"][i]) == list, "Not list"
						for k in range(len(data["bounding_box_augmenters"][i])):
							# Extract information
							augmentationType = list(data["bounding_box_augmenters"][i][k].keys())[0]
							if (not jsonConf.isValidBoundingBoxAugmentation(augmentation = augmentationType)):
								raise Exception("ERROR: {} is not valid.".format(augmentationType))
							parameters = data["bounding_box_augmenters"][i][k][augmentationType]
							# Save?
							saveParameter = self.extractSavingParameter(parameters = parameters)
							frame, bndboxes = self.__applyBoundingBoxAugmentation__(frame = frame,
																						boundingBoxes = bndboxes,
																						augmentationType = augmentationType, #j,
																						parameters = parameters)
							if (saveParameter == True):
								ImageLocalizationDataset.save_img_and_xml(frame = frame,
														bndboxes = bndboxes,
														names = names,
														database_name = self.databaseName,
														data_augmentation_type = augmentationType,
														origin_information = imgFullPath,
														output_image_directory = outputImageDirectory,
														output_annotation_directory = outputAnnotationDirectory)
					else:
						parameters = data["bounding_box_augmenters"][i]
						# Save?
						saveParameter = self.extractSavingParameter(parameters = parameters)
						frame, bndboxes = self.__applyBoundingBoxAugmentation__(frame = cv2.imread(imgFullPath),
																						boundingBoxes = boundingBoxes,
																						augmentationType = i,
																						parameters = parameters)
						# Save frame
						if (saveParameter == True):
							ImageLocalizationDataset.save_img_and_xml(frame = frame,
														bndboxes = bndboxes,
														names = names,
														database_name = self.databaseName,
														data_augmentation_type = i,
														origin_information = imgFullPath,
														output_image_directory = outputImageDirectory,
														output_annotation_directory = outputAnnotationDirectory)
			elif (typeAugmentation == 1):
				# Geometric data augmentations
				raise ValueError("Image geometric data augmentations are not " +\
													"supported for bounding boxes. Use bounding box " +\
													"augmentation types.")
			elif (typeAugmentation == 2):
				# Color data augmentations
				for i in data["image_color_augmenters"]:
					if (i == "Sequential"):
						# Prepare data for sequence
						frame = cv2.imread(imgFullPath)
						# Read elements of vector
						assert type(data["image_color_augmenters"][i]) == list, "Not list"
						for k in range(len(data["image_color_augmenters"][i])):
							# Extract information
							augmentationType = list(data["image_color_augmenters"][i][k].keys())[0]
							if (not jsonConf.isValidColorAugmentation(augmentation = augmentationType)):
								raise Exception("ERROR: {} is not valid.".format(augmentationType))
							parameters = data["image_color_augmenters"][i][k][augmentationType]
							# Save?
							saveParameter = self.extractSavingParameter(parameters = parameters)
							# Apply augmentation
							frame = self.__applyColorAugmentation__(frame = frame,
																						augmentationType = augmentationType, #j,
																						parameters = parameters)
							if (saveParameter == True):
								ImageLocalizationDataset.save_img_and_xml(frame = frame,
														bndboxes = boundingBoxes,
														names = names,
														database_name = self.databaseName,
														data_augmentation_type = augmentationType,
														origin_information = imgFullPath,
														output_image_directory = outputImageDirectory,
														output_annotation_directory = outputAnnotationDirectory)
					else:
						parameters = data["image_color_augmenters"][i]
						# Save?
						saveParameter = self.extractSavingParameter(parameters = parameters)
						frame = self.__applyColorAugmentation__(frame = cv2.imread(imgFullPath),
																						augmentationType = i,
																						parameters = parameters)
						# Save frame
						if (saveParameter == True):
							ImageLocalizationDataset.save_img_and_xml(frame = frame,
														bndboxes = boundingBoxes,
														names = names,
														database_name = self.databaseName,
														data_augmentation_type = i,
														origin_information = imgFullPath,
														output_image_directory = outputImageDirectory,
														output_annotation_directory = outputAnnotationDirectory)
			elif (typeAugmentation == 3):
				# Assert sequential follows multiple_image_augmentations
				if (not ("Sequential" in data["multiple_image_augmentations"])):
					raise Exception("ERROR: Data after multiple_image_augmentations is not recognized.")
				# Multiple augmentation configurations, get a list of hash maps of all the confs.
				list_of_augmenters_confs = data["multiple_image_augmentations"]["Sequential"]
				# Assert list_of_augmenters_confs is a list
				if (not (type(list_of_augmenters_confs) == list)):
					raise TypeError("ERROR: Data inside [multiple_image_augmentations][Sequential] must be a list.")
				# Prepare data for sequence.
				frame = cv2.imread(imgFullPath)
				bndboxes = boundingBoxes
				# print("\n*", list_of_augmenters_confs, "\n")
				for k in range(len(list_of_augmenters_confs)):
					# Get augmenter type ("bounding_box_augmenter" or "color_augmenter") position
					# in the list of multiple augmentations.
					augmentationConf = list(list_of_augmenters_confs[k].keys())[0]
					if (not (jsonConf.isBndBxAugConfFile(keys = [augmentationConf]) or
							jsonConf.isColorConfFile(keys = [augmentationConf]))):
						raise Exception("ERROR: {} is not a valid configuration.".format(augmentationConf))
					# Get sequential information from there. This information is a list of 
					# the types of augmenters that belong to augmentationConf.
					list_of_augmenters_confs_types = list_of_augmenters_confs[k][augmentationConf]["Sequential"]
					# Assert list_of_augmenters_confs is a list
					if (not (type(list_of_augmenters_confs_types) == list)):
						raise TypeError("ERROR: Data inside [multiple_image_augmentations][Sequential][{}][Sequential] must be a list."\
														.format(augmentationConf))
					# Iterate over augmenters inside sequential of type.
					for l in range(len(list_of_augmenters_confs_types)):
						# Get augmentation type and its parameters.
						augmentationType = list(list_of_augmenters_confs_types[l].keys())[0]
						# Assert augmentation is valid.
						if (not (jsonConf.isValidBoundingBoxAugmentation(augmentation = augmentationType) or
										jsonConf.isValidColorAugmentation(augmentation = augmentationType))):
							raise Exception("ERROR: {} is not valid.".format(augmentationType))
						parameters = list_of_augmenters_confs_types[l][augmentationType]
						# Save?
						saveParameter = self.extractSavingParameter(parameters = parameters)
						# Restart frame to original?
						restartFrameParameter = self.extractRestartFrameParameter(parameters = parameters)
						# Probability of augmentation happening.
						randomEvent = self.randomEvent(parameters = parameters, threshold = threshold)
						# print(augmentationType, parameters)
						# Apply augmentation
						if (augmentationConf == "image_color_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame = self.__applyColorAugmentation__(frame = frame,
																					augmentationType = augmentationType,
																					parameters = parameters)
						elif (augmentationConf == "bounding_box_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame, bndboxes = self.__applyBoundingBoxAugmentation__(frame = frame,
																					boundingBoxes = bndboxes,
																					augmentationType = augmentationType, #j,
																					parameters = parameters)
						# Save?
						if ((saveParameter == True) and (randomEvent == True)):
							ImageLocalizationDataset.save_img_and_xml(frame = frame,
								bndboxes = bndboxes,
								names = names,
								database_name = self.databaseName,
								data_augmentation_type = augmentationType,
								origin_information = imgFullPath,
								output_image_directory = outputImageDirectory,
								output_annotation_directory = outputAnnotationDirectory)
						# Restart frame?
						if (restartFrameParameter == True):
							frame = cv2.imread(imgFullPath)
							bndboxes = boundingBoxes
			else:
				raise Exception("Type augmentation {} not valid.".format(typeAugmentation))

	def extractSavingParameter(self, parameters = None):
		"""
		Extract the "save" parameter from a dictionary.
		Args:
			A dictionary.
		Returns:
			A boolean that contains the response of "save".
		"""
		if ("save" in parameters):
			if (type(parameters["save"]) != bool):
				raise TyperError("ERROR: Save parameter must be of type bool.")
			return parameters["save"]
		else:
			return False

	def extractRestartFrameParameter(self, parameters = None):
		"""
		Extracts the "restartFrame" parameter from a dictionary.
		Args:
			parameters: A dictionary.
		Returns:
			A boolean that contains the response of "restartFrame".
		"""
		if ("restartFrame" in parameters):
			if (type(parameters["restartFrame"]) != bool):
				raise TyperError("ERROR: Restart frame must be of type bool.")
			return parameters["restartFrame"]
		else:
			return False

	def randomEvent(self, parameters = None, threshold = None):
		"""
		Extracts the "randomEvent" parameter from a dictionary.
		Args:
			parameters: A dictionary.
			threshold: A float.
		Returns:
			A boolean that if true means the event should be executed.
		"""
		if ("randomEvent" in parameters):
			# Assert type.
			if (type(parameters["randomEvent"]) != bool):
				raise TyperError("ERROR: Random event must be of type bool.")
			# Check the value of randomEvent.
			if (parameters["randomEvent"] == True):
				activate = np.random.rand() > threshold
				# print(activate)
				return activate
			else:
				return True
		else:
			return True

	def __applyColorAugmentation__(self, frame = None, augmentationType = None, parameters = None):
		# Logic
		if (augmentationType == "invertColor"):
			frame = colorAugmenter.invertColor(frame = frame, CSpace = parameters["CSpace"])
		elif (augmentationType == "histogramEqualization"):
			frame = colorAugmenter.histogramEqualization(frame = frame, equalizationType = parameters["equalizationType"])
		elif (augmentationType == "changeBrightness"):
			frame = colorAugmenter.changeBrightness(frame = frame, coefficient = parameters["coefficient"])
		elif (augmentationType == "sharpening"):
			frame = colorAugmenter.sharpening(frame = frame, weight = parameters["weight"])
		elif (augmentationType == "addGaussianNoise"):
			frame = colorAugmenter.addGaussianNoise(frame = frame, coefficient = parameters["coefficient"])
		elif (augmentationType == "gaussianBlur"):
			frame = colorAugmenter.gaussianBlur(frame = frame, sigma = parameters["sigma"])
		elif (augmentationType == "shiftColors"):
			frame = colorAugmenter.shiftColors(frame = frame)
		elif (augmentationType == "fancyPCA"):
			frame = colorAugmenter.fancyPCA(frame = frame)
		else:
			raise Exception("Color augmentation type not supported: {}".format(augmentationType))
		# Return result
		return frame

	def __applyBoundingBoxAugmentation__(self, frame = None, boundingBoxes = None, augmentationType = None, parameters = None):
		# Local variables
		bndboxes = boundingBoxes
		# Logic
		if (augmentationType == "scale"):
			# Apply scaling
			if (not ("size" in parameters)):
				raise Exception("ERROR: Scale requires parameter size.")
			if (not ("zoom" in parameters)):
				raise Exception("ERROR: Scale requires parameter zoom.")
			if (not ("interpolationMethod" in parameters)):
				raise Exception("ERROR: Scale requires parameter interpolationMethod.")
			frame, bndboxes = bndboxAugmenter.scale(frame = frame,
										boundingBoxes = boundingBoxes,
										size = parameters["size"],
										zoom = parameters["zoom"],
										interpolationMethod = parameters["interpolationMethod"])
		elif (augmentationType == "crop"):
			# Apply crop
			if (not ("size" in parameters)):
				raise Exception("ERROR: Crop requires parameter size.")
			bndboxes = bndboxAugmenter.crop(boundingBoxes = boundingBoxes,
										size = parameters["size"])
		elif (augmentationType == "pad"):
			# Apply pad
			if (not ("size" in parameters)):
				raise Exception("ERROR: Pad requires parameter size.")
			bndboxes = bndboxAugmenter.pad(boundingBoxes = boundingBoxes,
																		frameHeight = frame.shape[0],
																		frameWidth = frame.shape[1],
																		size = parameters["size"])
		elif (augmentationType == "jitterBoxes"):
			# Apply jitter boxes
			if (not ("size" in parameters)):
				raise Exception("ERROR: JitterBoxes requires parameter size.")
			if (not ("quantity" in parameters)):
				raise Exception("ERROR: JitterBoxes requires parameter quantity.")
			frame = bndboxAugmenter.jitterBoxes(frame = frame,
																					boundingBoxes = boundingBoxes,
																					size = parameters["size"],
																					quantity = parameters["quantity"])
		elif (augmentationType == "horizontalFlip"):
			# Apply horizontal flip
			frame = bndboxAugmenter.horizontalFlip(frame = frame,
																						boundingBoxes = boundingBoxes)
		elif (augmentationType == "verticalFlip"):
			# Apply vertical flip
			frame = bndboxAugmenter.verticalFlip(frame = frame,
																					boundingBoxes = boundingBoxes)
		elif (augmentationType == "rotation"):
			# Apply rotation
			if (not ("theta" in parameters)):
				theta = None
				#raise Exception("ERROR: Rotation requires parameter theta.")
			else:
				theta = parameters["theta"]
			frame = bndboxAugmenter.rotation(frame = frame,
																				boundingBoxes = boundingBoxes,
																				theta = theta)
		elif (augmentationType == "dropout"):
			# Apply dropout
			if (not ("size" in parameters)):
				raise Exception("ERROR: Dropout requires parameter size.")
			if (not ("threshold" in parameters)):
				raise Exception("ERROR: Dropout requires parameter threshold.")
			frame = bndboxAugmenter.dropout(frame = frame,
																		boundingBoxes = boundingBoxes,
																		size = parameters["size"],
																		threshold = parameters["threshold"])
		return frame, bndboxes

	@staticmethod
	def save_img_and_xml(frame = None, bndboxes = None, names = None, database_name = None, data_augmentation_type = None, origin_information = None, output_image_directory = None, output_annotation_directory = None):
		"""
		Saves an image and its annotation.
		Args:
			frame: A numpy/tensorflow tensor that contains an image.
			bndboxes: A list of lists that contains the bounding boxes' coordinates.
			names: A list of strings that contains the labels of the bounding boxes.
			database_name: A string that contains the name of the database.
			data_augmentation_type: A string that contains the type of data augmentation.
			origin_information: A string that contains information about the file's 
													origin.
			output_image_directory: A string that contains the path to save the image.
			output_annotation_directory: A string that contains the path to save the 
																	image's annotation.
		Returns:
			None
		"""
		# Assertions
		if (database_name == None):
			database_name = "Unspecified"
		if (dataAssertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
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
		# Check file extension
		extension = Util.detect_file_extension(filename = origin_information)
		if (extension == None):
			raise Exception("Your image extension is not valid. Only jpgs and pngs are allowed. {}".format(extension))
		# Generate a new name.
		new_name = Util.create_random_name(name = database_name, length = 4)
		img_name = new_name + extension
		xml_name = new_name + ".xml"
		# Save image
		img_save_path = os.path.join(output_image_directory, img_name)
		cv2.imwrite(img_save_path, frame)
		output_annotation_directory = os.path.join(output_annotation_directory, \
																							xml_name)
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
		# Assert files have been written to disk.
		if (not os.path.isfile(output_annotation_directory)):
			print(origin_information)
			print(img_name)
			print(xml_name)
			raise Exception("ERROR: Annotation was not saved. This happens " +\
											"sometimes when there are dozens of thousands of data " +\
											"points. Please run the script again and report this problem.")
		if (not os.path.isfile(img_save_path)):
			raise Exception("ERROR: Image was not saved. This happens " +\
								"sometimes when there are dozens of thousands of data " +\
								"points. Please run the script again and report this problem.")

	@staticmethod
	def to_xml(filename = None, path = None, database_name = None, frame_size = None, data_augmentation_type = None, bounding_boxes = None, names = None, origin = None, output_directory = None):
		"""
		Creates an XML file that contains the annotation's information of an image.
		This file's structure is based on the VOC format.
		Args:
			filename: A string that contains the name of a file.
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

	@staticmethod
	def save_lists_in_dataframe(columns = None, data = None, output_directory = None):
		"""
		Save lists into a dataframe.
		Args:
			columns: A list of strings that contains the names of the columns 
							for the dataframe.
			data: A list of lists that contains the data for the dataframe.
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

class Annotation(object):
	def __init__(self, name = None, bndbox = None, module = None, corePoint = None):
		"""
		A class that holds parameters of a common annotation.
		Args:
			name: A string that contains a name.
			bndbox: A list of ints.
			module: A float.
			corePoint: A boolean.
		Returns:
			None
		"""
		super(Annotation, self).__init__()
		# Assertions
		if (name == None):
			raise ValueError("Name parameter cannot be empty.")
		if (bndbox == None):
			raise ValueError("Bounding box parameter cannot be empty.")
		if (module == None):
			module = -1
		if (corePoint == None):
			raise ValueError("corePoint parameter cannot be empty.")
		# Class variables
		self.name = name
		self.bndbox = bndbox
		self.module = module
		self.corePoint = corePoint
		self.otherAnnotations = []
		self.otherAnnotationsName = []

	@property
	def propertyModule(self):
		return self.module

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
	def propertyCorePoint(self):
		return self.corePoint

	@propertyCorePoint.setter
	def propertyCorePoint(self, corePoint):
		self.corePoint = corePoint

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
