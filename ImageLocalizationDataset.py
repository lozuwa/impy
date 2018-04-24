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

	def __init__(self,
							images = None,
							annotations = None,
							databaseName = None):
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
		imagesPath = [os.path.join(self.images, each) for each in \
									os.listdir(self.images)]
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
			imgFullPath = os.path.join(self.images, filename + extension)
			xmlFullPath = os.path.join(self.annotations, filename + ".xml")
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
		# Load image annotation
		annotation = ImageAnnotation(path = annotationPath)
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
		#   print(each.propertyName, each.propertyModule, each.propertyIndex)
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
						distance = VectorOperations.euclidean_distance(v0 = [ix0, iy0],
																														v1 = [x1, y1])
						if (distance < (offset-20)):
							annotations[i].includeOtherAnnotation([ix1, iy1, x1, y1])
							annotations[i].includeOtherAnnotationName(annotations[j].propertyName)
							annotations[j].propertyInUse = True

		# Debug
		# for each in annotations:
		#   print(each.propertyName, each.propertyIndex, \
		#         each.propertyOtherAnnotation, each.propertyOtherAnnotationName, "\n")
		# print("\n")

		# Save image croppings
		for i in range(len(annotations)):
			if (len(annotations[i].propertyOtherAnnotation) == 0):
				continue
			else:
				# Adjust image to current annotations' bounding boxes.
				RoiXMin, RoiYMin, \
				RoiXMax, RoiYMax = prep.adjustImage(frameHeight = height,
																		frameWidth = width,
																		boundingBoxes = annotations[i].propertyOtherAnnotation,
																		offset = offset)
				# Include bounding boxes after adjusting the region of interest.
				newBoundingBoxes,\
				newNames = prep.includeBoundingBoxes(edges = [RoiXMin, RoiYMin, RoiXMax, RoiYMax],
																						boundingBoxes = boundingBoxes,
																						names = names)
				# print((RoiXMax-RoiXMin), (RoiYMax-RoiYMin))
				# Read image
				frame = cv2.imread(imagePath)
				# Save image
				ImageLocalizationDataset.save_img_and_xml(frame = frame[RoiYMin:RoiYMax,\
																															RoiXMin:RoiXMax, :],
											bndboxes = newBoundingBoxes, #bdxs,
											names = newNames, #annotations[i].propertyOtherAnnotationName,
											database_name = self.databaseName,
											data_augmentation_type = "Unspecified",
											origin_information = imagePath,
											output_image_directory = outputImageDirectory,
											output_annotation_directory = outputAnnotationDirectory)

	def applyDataAugmentation(self, configurationFile = None, outputImageDirectory = None, outputAnnotationDirectory = None):
		"""
		Applies one or multiple data augmentation methods to the dataset.
		Args:
			configurationFile: A string with a path to a json file that contains the 
								configuration of the data augmentation methods.
			outputImageDirectory: A string that contains the path to the directory where
														images will be saved.
			outputAnnotationDirectory: A string that contains the path the directory where
																annotations will be saved.
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
		# Load configuration data.
		f = open(configurationFile)
		data = json.load(f)
		f.close()
		# Iterate over the images.
		for img in tqdm(os.listdir(self.images)[:1]):
			# Get the extension
			extension = Util.detect_file_extension(filename = img)
			if (extension == None):
				raise Exception("ERROR: Your image extension is not valid." +\
												 "Only jpgs and pngs are allowed.")
			# Extract name.
			filename = os.path.split(img)[1].split(extension)[0]
			# Create xml and img name.
			imgFullPath = os.path.join(self.images, filename + extension)
			xmlFullPath = os.path.join(self.annotations, filename + ".xml")
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
				# Multiple augmentation configurations, get a list of hash maps of all the confs.
				list_of_augmenters_confs = data["multiple_image_augmentations"]["Sequential"]
				# Prepare data for sequence.
				frame = cv2.imread(imgFullPath)
				bndboxes = boundingBoxes
				# print("\n*", list_of_augmenters_confs, "\n")
				for k in range(len(list_of_augmenters_confs)):
					# Get augmenter type ("bounding_box_augmenter" or "color_augmenter") position
					# in the list of multiple augmentations.
					augmentationConf = list(list_of_augmenters_confs[k].keys())[0]
					# Get sequential information from there. This information is a list of 
					# the types of augmenters that belong to augmentationConf.
					list_of_augmenters_confs_types = list_of_augmenters_confs[k][augmentationConf]["Sequential"]
					# Iterate over augmenters inside sequential of type.
					for l in range(len(list_of_augmenters_confs_types)):
						# Get augmentation type and its parameters.
						augmentationType = list(list_of_augmenters_confs_types[l].keys())[0]
						parameters = list_of_augmenters_confs_types[l][augmentationType]
						# Save?
						saveParameter = self.extractSavingParameter(parameters = parameters)            
						# print(augmentationType, parameters)
						# Apply augmentation
						if (augmentationConf == "image_color_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							frame = self.__applyColorAugmentation__(frame = frame,
																					augmentationType = augmentationType,
																					parameters = parameters)
							# Save frame
							if (saveParameter == True):
								ImageLocalizationDataset.save_img_and_xml(frame = frame,
															bndboxes = bndboxes,
															names = names,
															database_name = self.databaseName,
															data_augmentation_type = augmentationType,
															origin_information = imgFullPath,
															output_image_directory = outputImageDirectory,
															output_annotation_directory = outputAnnotationDirectory)
						elif (augmentationConf == "bounding_box_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							frame, bndboxes = self.__applyBoundingBoxAugmentation__(frame = frame,
																					boundingBoxes = bndboxes,
																					augmentationType = augmentationType, #j,
																					parameters = parameters)
							# Save frame
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
			return parameters["save"]
		else:
			return False

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
			frame, bndboxes = bndboxAugmenter.scale(frame = frame,
										boundingBoxes = boundingBoxes,
										size = parameters["size"],
										interpolationMethod = parameters["interpolationMethod"])
		elif (augmentationType == "crop"):
			# Apply crop
			bndboxes = bndboxAugmenter.crop(boundingBoxes = boundingBoxes,
										size = parameters["size"])
		elif (augmentationType == "pad"):
			# Apply pad
			bndboxes = bndboxAugmenter.pad(boundingBoxes = boundingBoxes,
																		frameHeight = frame.shape[0],
																		frameWidth = frame.shape[1],
																		size = parameters["size"])
		elif (augmentationType == "jitterBoxes"):
			# Apply jitter boxes
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
			frame = bndboxAugmenter.rotation(frame = frame,
																				boundingBoxes = boundingBoxes,
																				theta = parameters["theta"])
		elif (augmentationType == "dropout"):
			# Apply dropout
			frame = bndboxAugmenter.dropout(frame = frame,
																		boundingBoxes = boundingBoxes,
																		size = parameters["size"],
																		threshold = parameters["threshold"])
		return frame, bndboxes

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
		extension = Util.detect_file_extension(filename)
		if (extension == None):
			raise Exception("Image's extension not supported {}".format(filename))
		tree.write(os.path.join(output_directory, filename.split(extension)[0]+".xml"))

class Annotation(object):
	def __init__(self, 
							index = None,
							name = None,
							bndbox = None,
							module = None,
							inUse = None):
		super(Annotation, self).__init__()
		# Assertions
		if (index == None):
			raise ValueError("Index parameter cannot be empty.")
		if (name == None):
			raise ValueError("Name parameter cannot be empty.")
		if (bndbox == None):
			raise ValueError("Bounding box parameter cannot be empty.")
		if (module == None):
			module = -1
		if (inUse == None):
			raise ValueError("InUse parameter cannot be empty.")
		# Class variables
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