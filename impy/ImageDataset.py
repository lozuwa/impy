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
	from .AugmentationConfigurationFile import *
except:
	from AugmentationConfigurationFile import *

try:
	from .ApplyAugmentation import applyGeometricAugmentation, applyColorAugmentation
except:
	from ApplyAugmentation import applyGeometricAugmentation, applyColorAugmentation

try:
	from .Util import *
except:
	from Util import *

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
			raise TypeError("ERROR: threshold parameter must be of type float.")
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
				raise Exception("Bounding box augmenters cannot be applied to an image dataset." +\
												" Use geometric augmenters instead.")
			elif (typeAugmentation == 1):
				# Geometric data augmentations
				for i in data["image_geometric_augmenters"]:
					if (i == "Sequential"):
						# Prepare data for sequence
						frame = cv2.imread(imgFullPath)
						# Read elements of vector
						assert type(data["image_geometric_augmenters"][i]) == list, "Not list"
						for k in range(len(data["image_geometric_augmenters"][i])):
							# Extract information
							augmentationType = list(data["image_geometric_augmenters"][i][k].keys())[0]
							if (not jsonConf.isValidGeometricAugmentation(augmentation = augmentationType)):
								raise Exception("ERROR: {} is not valid.".format(augmentationType))
							parameters = data["image_geometric_augmenters"][i][k][augmentationType]
							# Save?
							saveParameter = jsonConf.extractSavingParameter(parameters = parameters)
							# Apply augmentation
							frame = applyColorAugmentation(frame = frame,
																						augmentationType = augmentationType, #j,
																						parameters = parameters)
							if (saveParameter == True):
								# Generate a new name.
								newName = Util.create_random_name(name = self.dbName, length = 4)
								imgName = newName + extension
								# Save image.
								Util.save_img(frame = frame, 
															img_name = imgName, 
															output_image_directory = outputImageDirectory)
					else:
						parameters = data["image_geometric_augmenters"][i]
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
						# Apply augmentation.
						if (augmentationConf == "image_color_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame = applyColorAugmentation(frame = frame,
																					augmentationType = augmentationType,
																					parameters = parameters)
						elif (augmentationConf == "image_geometric_augmenters"):
							# print(augmentationConf, augmentationType, parameters)
							if (randomEvent == True):
								frame = applyGeometricAugmentation(frame = frame,
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

		