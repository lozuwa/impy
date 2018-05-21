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
from interface import implements
from tqdm import tqdm

try:
	from .ImageLocalizationDatasetPreprocessMethods import *
except:
	from ImageLocalizationDatasetPreprocessMethods import *

try:
	from .ImageLocalizationDatasetStatisticsMethods import *
except:
	from ImageLocalizationDatasetStatisticsMethods import *

try:
	from .ImagePreprocess import *
except:
	from ImagePreprocess import *

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
	from .AugmentationConfigurationFile import *
except:
	from AugmentationConfigurationFile import *

try:
	from .ApplyAugmentation import applyBoundingBoxAugmentation, applyColorAugmentation
except:
	from ApplyAugmentation import applyBoundingBoxAugmentation, applyColorAugmentation

prep = ImagePreprocess()
dataAssertion = AssertDataTypes()

class ImageLocalizationDataset(implements(ImageLocalizationDatasetPreprocessMethods, \
																ImageLocalizationDatasetStatisticsMethods)):

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

	# Preprocessing.
	def dataConsistency(self):
		"""
		Checks whether data is consistent. It starts analyzing if there is the same amount of 
		of images and annotations. Then it sees if the annotations and images are consistent 
		with each other.
		Args:
			None
		Returns:
			None
		Raises:
			- Exception: when the extension of the image is not allowed. Only jpgs and pngs are allowed.
			- Exception: When an annotation file does not have a .xml extension.
			- Exception: When the amount of annotations and images is not equal.
			- Exception: When there are images that don't have annotations.
			- Exception: When there are annotations that don't have images.
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
		Raises:
			- Exception: when the extension of the image is not allowed. Only jpgs and pngs are allowed.
			- Exception: when an annotation file is empty.
			- Exception: when a coordinate is not valid. Either less than zero or greater than image's size.
		"""
		# Assertions
		if (removeEmpty == None):
			removeEmpty = False
		# Local variables
		emptyAnnotations = []
		files = os.listdir(self.imagesDirectory)
		# Logic
		for file in tqdm(files):
			# In case a folder is found, report it.
			if (os.path.isdir(file)):
				continue
			# Otherwise, continue.
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
					#os.remove(imgFullPath)
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

	# Stats.
	def computeBoundingBoxStats(self, saveDataFrame = None, outputDirDataFrame = None):
		"""
		Compute basic stats for the dataset's bounding boxes.
		Args:
			saveDataFrame: A boolean that defines whether to save the dataframe or not.
			outputDirDataFrame: A string that contains the path where the dataframe will
													be saved.
		Returns:
			None
		"""
		# Assertions
		if (saveDataFrame == None):
			saveDataFrame = False
		else:
			if (type(saveDataFrame) == bool):
				if (outputDirDataFrame == None):
					raise ValueError("Parameter directory dataframe cannot be empty.")
			else:
				raise TypeError("saveDataFrame must be of type bool.")
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
			Util.save_lists_in_dataframe(columns = columns,
									data = [paths, names, widths, heights, boundingBoxesLists],
									output_directory = outputDirDataFrame)

	# Save bounding boxes as files.
	def saveBoundingBoxes(self, outputDirectory = None, filterClasses = None):
		"""
		Saves the bounding boxes as images of each image in the dataset.
		Args:
			outputDirectory: A string that contains the directory where the images will be saved.
			filterClasses: A list of Strings that contains names of the classes to be filtered and saved.
		Returns:
			None
		"""
		# Assertions
		if (outputDirectory == None):
			raise ValueError("outputDirectory cannot be empty")
		if (type(outputDirectory) != str):
			raise TyperError("outputDirectory must be a string.")
		if (not (os.path.isdir(outputDirectory))):
			raise FileNotFoundError("outputDirectory's path does not exist: ".format(outputDirectory))
		if (filterClasses == None):
			filterClasses = []
		if (type(filterClasses) != list):
			raise TyperError("filterClasses must be of type list.")
		# Local variables
		images = [os.path.join(self.imagesDirectory, i) for i in os.listdir(self.imagesDirectory)]
		# Logic
		for img in tqdm(images):
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
			# Load annotation.
			annt = ImageAnnotation(path = xmlFullPath)
			# Get bounding boxes.
			boundingBoxes = annt.propertyBoundingBoxes
			names = annt.propertyNames
			# Save image.
			frame = cv2.imread(img)
			# Save bounding boxes as png images.
			for name, boundingBox in zip(names, boundingBoxes):
				if ((len(filterClasses) == 0) or (name in filterClasses)):
					ix, iy, x, y = boundingBox
					# Detect extension.
					extension = Util.detect_file_extension(filename = img)
					if (extension == None):
						raise Exception("Your image extension is not valid. " +\
														"Only jpgs and pngs are allowed. {}".format(extension))
					# Generate a new name.
					newName = Util.create_random_name(name = self.databaseName, length = 4)
					imgName = newName + extension
					# Check bounding box does not get out of boundaries.
					if (x == frame.shape[1]):
						x -= 1
					if (y == frame.shape[0]):
						y -= 1
					# Check bounding boxes are ok.
					if (((y-iy) == 0) or ((x - ix) == 0) or \
							((ix < 0) or (iy < 0)) or \
							((x > frame.shape[1]) or (y > frame.shape[0]))):
						print(img)
						print(ix, iy, x, y)
						raise Exception("Bounding box does not exist.")
					# Save image.
					Util.save_img(frame = frame[iy:y, ix:x, :],
																						img_name = imgName,
																						output_image_directory = outputDirectory)

	# Reduce and data augmentation.
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
			raise ValueError("Offset parameter cannot be empty.")
		if (outputImageDirectory == None):
			outputImageDirectory = os.getcwd()
			Util.create_folder(os.path.join(outputImageDirectory, "images"))
			outputImageDirectory = os.path.join(os.getcwd(), "images")
		if (not (os.path.isdir(outputImageDirectory))):
			raise Exception("Path to output directory does not exist. {}"\
											.format(outputImageDirectory))
		if (outputAnnotationDirectory == None):
			outputAnnotationDirectory = os.getcwd()
			Util.create_folder(os.path.join(outputAnnotationDirectory, "annotations"))
			Util.create_folder(os.path.join(outputAnnotationDirectory, "annotations", "xmls"))
			outputAnnotationDirectory = os.path.join(os.getcwd(), "annotations", "xmls")
		if (not (os.path.isdir(outputAnnotationDirectory))):
			raise Exception("Path to output annotation directory does not exist. {}"\
											.format(outputAnnotationDirectory))
		# Get images and annotations full paths
		imagesPath = [os.path.join(self.imagesDirectory, each) for each in \
									os.listdir(self.imagesDirectory)]
		for img in tqdm(imagesPath):
			#print(img)
			# Get extension
			extension = Util.detect_file_extension(filename = img)
			if (extension == None):
				raise Exception("Your image extension is not valid." +\
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
				# Read image.
				frame = cv2.imread(imagePath)
				extension = Util.detect_file_extension(filename = imagePath)
				if (extension == None):
					raise Exception("Your image extension is not valid. " +\
													"Only jpgs and pngs are allowed. {}".format(extension))
				# Generate a new name.
				newName = Util.create_random_name(name = self.databaseName, length = 4)
				imgName = newName + extension
				xmlName = newName + ".xml"
				# Save image.
				Util.save_img(frame = frame[RoiYMin:RoiYMax, RoiXMin:RoiXMax, :],
																					img_name = imgName,
																					output_image_directory = outputImageDirectory)
				# Save annotation.
				Util.save_annotation(filename = imgName,
													path = os.path.join(outputImageDirectory, imgName),
													database_name = self.databaseName,
													frame_size = frame[RoiYMin:RoiYMax, RoiXMin:RoiXMax, :].shape,
													data_augmentation_type = "Unspecified",
													bounding_boxes = newBoundingBoxes,
													names = newNames,
													origin = imagePath,
													output_directory = os.path.join(outputAnnotationDirectory, xmlName))

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
		jsonConf = AugmentationConfigurationFile(file = configurationFile)
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
							saveParameter = jsonConf.extractSavingParameter(parameters = parameters)
							frame, bndboxes = applyBoundingBoxAugmentation(frame = frame,
																						boundingBoxes = bndboxes,
																						augmentationType = augmentationType, #j,
																						parameters = parameters)
							if (saveParameter == True):
								# Generate a new name.
								newName = Util.create_random_name(name = self.databaseName, length = 4)
								imgName = newName + extension
								xmlName = newName + ".xml"
								# Save image.
								Util.save_img(frame = frame, 
																									img_name = imgName, 
																									output_image_directory = outputImageDirectory)
								# Save annotation.
								Util.save_annotation(filename = imgName,
																						path = os.path.join(outputImageDirectory, imgName),
																						database_name = self.databaseName,
																						frame_size = frame.shape,
																						data_augmentation_type = augmentationType,
																						bounding_boxes = bndboxes,
																						names = names,
																						origin = imgFullPath,
																						output_directory = os.path.join(outputAnnotationDirectory, xmlName))
					else:
						parameters = data["bounding_box_augmenters"][i]
						# Save?
						saveParameter = jsonConf.extractSavingParameter(parameters = parameters)
						frame, bndboxes = applyBoundingBoxAugmentation(frame = cv2.imread(imgFullPath),
																						boundingBoxes = boundingBoxes,
																						augmentationType = i,
																						parameters = parameters)
						# Save frame
						if (saveParameter == True):
							# Generate a new name.
							newName = Util.create_random_name(name = self.databaseName, length = 4)
							imgName = newName + extension
							xmlName = newName + ".xml"
							# Save image.
							Util.save_img(frame = frame, 
																								img_name = imgName, 
																								output_image_directory = outputImageDirectory)
							# Save annotation.
							Util.save_annotation(filename = imgName,
																					path = os.path.join(outputImageDirectory, imgName),
																					database_name = self.databaseName,
																					frame_size = frame.shape,
																					data_augmentation_type = augmentationType,
																					bounding_boxes = bndboxes,
																					names = names,
																					origin = imgFullPath,
																					output_directory = os.path.join(outputAnnotationDirectory, xmlName))
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
							saveParameter = jsonConf.extractSavingParameter(parameters = parameters)
							# Apply augmentation
							frame = applyColorAugmentation(frame = frame,
																						augmentationType = augmentationType, #j,
																						parameters = parameters)
							if (saveParameter == True):
								# Generate a new name.
								newName = Util.create_random_name(name = self.databaseName, length = 4)
								imgName = newName + extension
								xmlName = newName + ".xml"
								# Save image.
								Util.save_img(frame = frame, 
																									img_name = imgName, 
																									output_image_directory = outputImageDirectory)
								# Save annotation.
								Util.save_annotation(filename = imgName,
																						path = os.path.join(outputImageDirectory, imgName),
																						database_name = self.databaseName,
																						frame_size = frame.shape,
																						data_augmentation_type = augmentationType,
																						bounding_boxes = bndboxes,
																						names = names,
																						origin = imgFullPath,
																						output_directory = os.path.join(outputAnnotationDirectory, xmlName))
					else:
						parameters = data["image_color_augmenters"][i]
						# Save?
						saveParameter = jsonConf.extractSavingParameter(parameters = parameters)
						frame = applyColorAugmentation(frame = cv2.imread(imgFullPath),
																						augmentationType = i,
																						parameters = parameters)
						# Save frame
						if (saveParameter == True):
							# Generate a new name.
							newName = Util.create_random_name(name = self.databaseName, length = 4)
							imgName = newName + extension
							xmlName = newName + ".xml"
							# Save image.
							Util.save_img(frame = frame, 
																								img_name = imgName, 
																								output_image_directory = outputImageDirectory)
							# Save annotation.
							Util.save_annotation(filename = imgName,
																					path = os.path.join(outputImageDirectory, imgName),
																					database_name = self.databaseName,
																					frame_size = frame.shape,
																					data_augmentation_type = augmentationType,
																					bounding_boxes = bndboxes,
																					names = names,
																					origin = imgFullPath,
																					output_directory = os.path.join(outputAnnotationDirectory, xmlName))
			elif (typeAugmentation == 3):
				# Assert sequential follows multiple_image_augmentations.
				if (not ("Sequential" in data["multiple_image_augmentations"])):
					raise Exception("ERROR: Data after multiple_image_augmentations is not recognized.")
				# Multiple augmentation configurations, get a list of hash maps of all the confs.
				list_of_augmenters_confs = data["multiple_image_augmentations"]["Sequential"]
				# Assert list_of_augmenters_confs is a list.
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
						raise Exception("{} is not a valid configuration.".format(augmentationConf))
					# Get sequential information from there. This information is a list of 
					# the types of augmenters that belong to augmentationConf.
					list_of_augmenters_confs_types = list_of_augmenters_confs[k][augmentationConf]["Sequential"]
					# Assert list_of_augmenters_confs is a list
					if (not (type(list_of_augmenters_confs_types) == list)):
						raise TypeError("Data inside [multiple_image_augmentations][Sequential][{}][Sequential] must be a list."\
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
						saveParameter = jsonConf.extractSavingParameter(parameters = parameters)
						# Restart frame to original?
						restartFrameParameter = jsonConf.extractRestartFrameParameter(parameters = parameters)
						# Probability of augmentation happening.
						randomEvent = jsonConf.randomEvent(parameters = parameters, threshold = threshold)
						# print(augmentationType, parameters)
						# Apply augmentation.
						if (augmentationConf == "image_color_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame = applyColorAugmentation(frame = frame,
																					augmentationType = augmentationType,
																					parameters = parameters)
						elif (augmentationConf == "bounding_box_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame, bndboxes = applyBoundingBoxAugmentation(frame = frame,
																					boundingBoxes = bndboxes,
																					augmentationType = augmentationType, #j,
																					parameters = parameters)
						# Save?
						if ((saveParameter == True) and (randomEvent == True)):
							# Generate a new name.
							newName = Util.create_random_name(name = self.databaseName, length = 4)
							imgName = newName + extension
							xmlName = newName + ".xml"
							# Save image.
							Util.save_img(frame = frame, 
														img_name = imgName, 
														output_image_directory = outputImageDirectory)
							# Save annotation.
							Util.save_annotation(filename = imgName,
																	path = os.path.join(outputImageDirectory, imgName),
																	database_name = self.databaseName,
																	frame_size = frame.shape,
																	data_augmentation_type = augmentationType,
																	bounding_boxes = bndboxes,
																	names = names,
																	origin = imgFullPath,
																	output_directory = os.path.join(outputAnnotationDirectory, xmlName))
						# Restart frame?
						if (restartFrameParameter == True):
							frame = cv2.imread(imgFullPath)
							bndboxes = boundingBoxes
			else:
				raise Exception("Type augmentation {} not valid.".format(typeAugmentation))

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
