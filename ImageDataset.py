"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: A class that loads an image dataset and performs
useful operations with it.
"""
import os
import json
import math
import numpy as np
from interface import implements
from tqdm import tqdm

try:
	from .GeometricAugmenters import *
except:
	from GeometricAugmenters import *

try:
	from .ColorAugmenters import *
except:
	from ColorAugmenters import *

try:
	from .AugmentationConfigurationFile import *
except:
	from AugmentationConfigurationFile import *

try:
	from .Util import *
except:
	from Util import *

geometricAugmenter = GeometricAugmenters()
colorAugmenter = ColorAugmenters()

class ImageDataset(object):
	def __init__(self, imagesDirectory = None, dbName = None):
		super(ImageDataset, self).__init__()
		# Assertions.
		if (imagesDirectory == None):
			raise Exception("imagesDirectory cannot be empty.")
		if (type(imagesDirectory) != str):
			raise TypeError("imagesDirectory should be of type str.")
		if (not os.path.isdir(imagesDirectory)):
			raise Exception("imagesDirectory's path does not exist: {}"\
											.format(imagesDirectory))
		if (dbName == None):
			dbName = "Unspecified"
		if (type(dbName) != str):
			raise TypeError("dbName must be of type string.")
		# Class variables.
		self.imagesDirectory = imagesDirectory
		self.dbName = dbName

	def applyDataAugmentation(self, configurationFile = None, outputImageDirectory = None, threshold = None):
		"""
		Applies one or multiple data augmentation methods to the dataset.
		Args:
			configurationFile: A string with a path to a json file that contains the 
								configuration of the data augmentation methods.
			outputImageDirectory: A string that contains the path to the directory where
														images will be saved.
			threshold: A float that contains a number between 0 and 1.
		Returns:
			None
		"""
		# Assertions 
		if (configurationFile == None):
			raise FileNotFoundError("configuration file's path has not been found.")
		else:
			if (not os.path.isfile(configurationFile)):
				raise FileNotFoundError("Path to json file ({}) does not exist."\
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
			# Get the extension.
			extension = Util.detect_file_extension(filename = img)
			if (extension == None):
				raise Exception("Your image extension is not valid." +\
												 "Only jpgs and pngs are allowed.")
			# Extract name.
			filename = os.path.split(img)[1].split(extension)[0]
			# Create xml and img name.
			imgFullPath = os.path.join(self.imagesDirectory, filename + extension)
			# Apply augmentation.
			if (typeAugmentation == 0):
				for i in data["color_augmenters"]:
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
							frame, bndboxes = self.__applyBoundingBoxAugmentation__(frame = frame,
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
						frame, bndboxes = self.__applyBoundingBoxAugmentation__(frame = cv2.imread(imgFullPath),
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
							frame = self.__applyColorAugmentation__(frame = frame,
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
						frame = self.__applyColorAugmentation__(frame = cv2.imread(imgFullPath),
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
				# Assert sequential follows multiple_image_augmentations
				if (not ("Sequential" in data["multiple_image_augmentations"])):
					raise Exception("Data after multiple_image_augmentations is not recognized.")
				# Multiple augmentation configurations, get a list of hash maps of all the confs.
				list_of_augmenters_confs = data["multiple_image_augmentations"]["Sequential"]
				# Assert list_of_augmenters_confs is a list
				if (not (type(list_of_augmenters_confs) == list)):
					raise TypeError("Data inside [multiple_image_augmentations][Sequential] must be a list.")
				# Prepare data for sequence.
				frame = cv2.imread(imgFullPath)
				# print("\n*", list_of_augmenters_confs, "\n")
				for k in range(len(list_of_augmenters_confs)):
					# Get augmenter type ("image_geometric_augmenters" or "image_color_augmenters") position
					# in the list of multiple augmentations.
					augmentationConf = list(list_of_augmenters_confs[k].keys())[0]
					if (not (jsonConf.isGeometricConfFile(keys = [augmentationConf]) or
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
						if (not (jsonConf.isValidGeometricAugmentation(augmentation = augmentationType) or
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
						# Apply augmentation
						if (augmentationConf == "image_color_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame = self.__applyColorAugmentation__(frame = frame,
																					augmentationType = augmentationType,
																					parameters = parameters)
						elif (augmentationConf == "image_geometric_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame = self.__applyGeometricAugmentation__(frame = frame,
																					augmentationType = augmentationType,
																					parameters = parameters)
						# Save?
						if ((saveParameter == True) and (randomEvent == True)):
							# Generate a new name.
							newName = Util.create_random_name(name = self.dbName, length = 4)
							imgName = newName + extension
							# Save image.
							Util.save_img(frame = frame, 
														img_name = imgName, 
														output_image_directory = outputImageDirectory)
						# Restart frame?
						if (restartFrameParameter == True):
							frame = cv2.imread(imgFullPath)
			else:
				raise Exception("Type augmentation {} not valid.".format(typeAugmentation))

	def __applyColorAugmentation__(self, frame = None, augmentationType = None, parameters = None):
		# Logic
		if (augmentationType == "invertColor"):
			if (not ("CSpace" in parameters)):
				parameters["CSpace"] = [True, True, True]
			frame = colorAugmenter.invertColor(frame = frame, CSpace = parameters["CSpace"])
		elif (augmentationType == "histogramEqualization"):
			if (not ("equalizationType" in parameters)):
				parameters["equalizationType"] = 0
			frame = colorAugmenter.histogramEqualization(frame = frame, equalizationType = parameters["equalizationType"])
		elif (augmentationType == "changeBrightness"):
			if (not ("coefficient" in parameters)):
				raise AttributeError("coefficient for changeBrightness must be specified.")
			frame = colorAugmenter.changeBrightness(frame = frame, coefficient = parameters["coefficient"])
		elif (augmentationType == "sharpening"):
			if (not ("weight" in parameters)):
				raise AttributeError("weight for sharpening must be specified.")
			frame = colorAugmenter.sharpening(frame = frame, weight = parameters["weight"])
		elif (augmentationType == "addGaussianNoise"):
			if (not ("coefficient" in parameters)):
				raise AttributeError("coefficient for addGaussianNoise must be specified.")
			frame = colorAugmenter.addGaussianNoise(frame = frame, coefficient = parameters["coefficient"])
		elif (augmentationType == "gaussianBlur"):
			if (not ("sigma" in parameters)):
				raise AttributeError("sigma for gaussianBlur must be specified.")
			frame = colorAugmenter.gaussianBlur(frame = frame, sigma = parameters["sigma"])
		elif (augmentationType == "shiftColors"):
			frame = colorAugmenter.shiftColors(frame = frame)
		elif (augmentationType == "fancyPCA"):
			frame = colorAugmenter.fancyPCA(frame = frame)
		else:
			raise Exception("Color augmentation type not supported: {}".format(augmentationType))
		# Return result
		return frame

	def __applyGeometricAugmentation__(self, frame = None, augmentationType = None, parameters = None):
		# Logic
		if (augmentationType == "scale"):
			# Apply scaling
			if (not ("size" in parameters)):
				raise Exception("ERROR: Scale requires parameter size.")
			if (not ("interpolationMethod" in parameters)):
				raise Exception("ERROR: Scale requires parameter interpolationMethod.")
			frame = geometricAugmenter.scale(frame = frame,
										size = parameters["size"],
										interpolationMethod = parameters["interpolationMethod"])
		elif (augmentationType == "crop"):
			# Apply crop
			if (not ("size" in parameters)):
				parameters["size"] = [0,0]
			frame = geometricAugmenter.crop(frame = frame,
																	size = parameters["size"])
		elif (augmentationType == "translate"):
			# Apply pad
			if (not ("offset" in parameters)):
				raise Exception("ERROR: Pad requires parameter offset.")
			bndboxes = geometricAugmenter.translate(frame = frame,
																		offset = parameters["offset"])
		elif (augmentationType == "jitterBoxes"):
			# Apply jitter boxes
			if (not ("size" in parameters)):
				raise Exception("ERROR: JitterBoxes requires parameter size.")
			if (not ("quantity" in parameters)):
				raise Exception("ERROR: JitterBoxes requires parameter quantity.")
			if (not ("color" in parameters)):
				parameters["color"] = [255,255,255]
			frame = geometricAugmenter.jitterBoxes(frame = frame,
																					size = parameters["size"],
																					quantity = parameters["quantity"],
																					color = parameters["color"])
		elif (augmentationType == "horizontalFlip"):
			# Apply horizontal flip
			frame = geometricAugmenter.horizontalFlip(frame = frame)
		elif (augmentationType == "verticalFlip"):
			# Apply vertical flip
			frame = geometricAugmenter.verticalFlip(frame = frame)
		elif (augmentationType == "rotation"):
			# Apply rotation
			if (not ("theta" in parameters)):
				theta = None
				#raise Exception("ERROR: Rotation requires parameter theta.")
			else:
				theta = parameters["theta"]
			frame = geometricAugmenter.rotation(frame = frame,
																				bndbox = [0, 0, frame.shape[1], frame.shape[0]],
																				theta = theta)
		return frame
		