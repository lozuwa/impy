import os
import json
import numpy as np
from interface import implements

class AugmentationConfigurationFile(object):
	def __init__(self, file = None):
		super(AugmentationConfigurationFile, self).__init__()
		# Assertions
		if (file == None):
			raise ValueError("File parameter cannot be None.")
		if (not (os.path.isfile(file))):
			raise Exception("Path to {} does not exist.".format(file))
		if (not (file.endswith("json"))):
			raise Exception("Configuration file has to be of JSON format.")
		# Class variables
		f = open(file)
		self.file = json.load(f)
		f.close()
		# Hardcoded configurations
		# Types of data augmenters
		self.confAugBndbxs = "bounding_box_augmenters"
		self.confAugGeometric = "image_geometric_augmenters"
		self.confAugColor = "image_color_augmenters"
		self.confAugMultiple = "multiple_image_augmentations"
		self.supportedDataAugmentationTypes = [self.confAugMultiple, self.confAugBndbxs, self.confAugGeometric, self.confAugColor]
		# Augmenter methods for bounding boxes
		self.scale = "scale"
		self.crop = "crop"
		self.pad = "pad"
		self.jitterBoxes = "jitterBoxes"
		self.horizontalFlip = "horizontalFlip"
		self.verticalFlip = "verticalFlip"
		self.rotation = "rotation"
		self.dropout = "dropout"
		self.boundingBoxesMethods = [self.scale, self.crop, self.pad, self.jitterBoxes, \
																self.horizontalFlip, self.verticalFlip, self.rotation, \
																self.dropout]
		self.invertColor = "invertColor"
		self.histogramEqualization = "histogramEqualization"
		self.changeBrightness = "changeBrightness"
		self.sharpening = "sharpening"
		self.addGaussianNoise = "addGaussianNoise"
		self.gaussianBlur = "gaussianBlur"
		self.shiftColors = "shiftColors"
		self.fancyPCA = "fancyPCA"
		self.colorMethods = [self.invertColor, self.histogramEqualization, self.changeBrightness, \
													self.sharpening, self.addGaussianNoise, self.gaussianBlur, \
													self.shiftColors, self.fancyPCA]

	def isValidBoundingBoxAugmentation(self, augmentation = None):
		"""
		Asserts that augmentation is a valid bounding box augmentation supported by the library.
		Args:
			augmentation: A string that contains an augmentation type.
		Returns:
			A boolean that if true means the augmentation type is valid, 
			otherwise it is false.
		"""
		# Assertions
		if (augmentation == None):
			raise ValueError("ERROR: augmentation parameter cannot be empty." +\
											" Report this problem.")
		# Logic
		if (augmentation in self.boundingBoxesMethods):
			return True
		else:
			return False

	def isValidColorAugmentation(self, augmentation = None):
		"""
		Asserts that augmentation is a valid color augmentation supported by the library.
		Args:
			augmentation: A string that contains an augmentation type.
		Returns:
			A boolean that if true means the augmentation type is valid, 
			otherwise it is false.
		"""
		# Assertions
		if (augmentation == None):
			raise ValueError("ERROR: augmentation parameter cannot be empty." +\
											" Report this problem.")
		# Logic
		if (augmentation in self.colorMethods):
			return True
		else:
			return False

	def runAllAssertions(self):
		"""
		Macro function that runs multiple assertions in order to validate 
		the configuration file.
		"""
		# Get keys
		keys = [i for i in self.file.keys()]
		# Run assertions
		self.isValidConfFile(keys = keys)
		rBndbx = self.isBndBxAugConfFile(keys = keys)
		rGeometric = self.isGeometricConfFile(keys = keys)
		rColor = self.isColorConfFile(keys = keys)
		rMultiple = self.isMultipleConfFile(keys = keys)
		# Return type of augmentation.
		if (rBndbx):
			return 0
		elif (rGeometric):
			return 1
		elif (rColor):
			return 2
		elif (rMultiple):
			return 3
		else:
			raise Exception("The configuration is not valid: {}.".format(keys) +\
							"bndbx: {} geometric: {} color: {}".format(rBndbx, rGeometric, rColor))

	def isValidConfFile(self, keys = None):
		# Check len of keys
		self.lenOfKeys(keys = keys)
		# Check key's values are supported.
		self.isKeyValid(keys = keys)

	def lenOfKeys(self, keys = None):
		"""
		Check the len of keys contained in keys. If the len is distinct of 1, then
		raise an error.
		Args:
			keys: A list of strings.
		Returns:
			None
		Raises: If the len of keys is different of 1. 
		"""
		# Assertions
		if (keys == None):
			raise ValueError("ERROR: Keys parameter cannot be empty.")
		if (type(keys) != list):
			raise ValueError("ERROR: keys should be a list.")
		# Check amount of configurations.
		if (len(keys) != 1):
			raise Exception("ERROR: Configuration file cannot have more than 1 configuration." + \
											"These were found: {}".format(keys))

	def isKeyValid(self, keys = None):
		"""
		Args:
			keys: A list of strings.
		Returns:
			None
		Raises:
			If the key in keys is not supported.
		"""
		# Assertions
		if (keys == None):
			raise ValueError("ERROR: Keys parameter cannot be empty.")
		if (type(keys) != list):
			raise ValueError("ERROR: keys should be a list.")
		if (keys[0] in self.supportedDataAugmentationTypes):
			pass
		else:
			raise Exception("Configuration type {} not supported.".format(keys[0]))

	def isBndBxAugConfFile(self, keys = None):
		"""
		Check if file is a bounding box configuration file.
		Args:
			keys: A list of strings.
		Returns:
			A boolean that is true if the conf file is bndboxes.
		"""
		# Assertions
		if (keys == None):
			raise ValueError("ERROR: Keys parameter cannot be empty.")
		if (type(keys) != list):
			raise ValueError("ERROR: keys should be a list.")
		if (len(keys) != 1):
			raise ValueError("ERROR: keys should be of len > 1.")
		# Check for type of configuration
		if (keys[0] == self.confAugBndbxs):
			return True
		else:
			return False

	def isGeometricConfFile(self, keys = None):
		"""
		Check if file is a bounding box configuration file.
		Args:
			keys: A list of strings.
		Returns:
			A boolean that is true if the conf file is geomtric.
		"""
		# Assertions
		if (keys == None):
			raise ValueError("ERROR: Keys parameter cannot be empty.")
		if (type(keys) != list):
			raise ValueError("ERROR: keys should be a list.")
		if (len(keys) != 1):
			raise ValueError("ERROR: keys should be of len > 1.")
		# Check for type of configuration
		if (keys[0] == self.confAugGeometric):
			return True
		else:
			return False

	def isColorConfFile(self, keys = None):
		"""
		Check if file is a bounding box configuration file.
		Args:
			keys: A list of strings.
		Returns:
			A boolean that is true if the conf file is color.
		"""
		# Assertions
		if (keys == None):
			raise ValueError("ERROR: Keys parameter cannot be empty.")
		if (type(keys) != list):
			raise ValueError("ERROR: keys should be a list.")
		if (len(keys) != 1):
			raise ValueError("ERROR: keys should be of len > 1.")
		# Check for type of configuration
		if (keys[0] == self.confAugColor):
			return True
		else:
			return False

	def isMultipleConfFile(self, keys = None):
		"""
		Check if file is a bounding box configuration file.
		Args:
			keys: A list of strings.
		Returns:
			A boolean that is true if the conf file is color.
		"""
		# Assertions
		if (keys == None):
			raise ValueError("ERROR: Keys parameter cannot be empty.")
		if (type(keys) != list):
			raise ValueError("ERROR: keys should be a list.")
		if (len(keys) != 1):
			raise ValueError("ERROR: keys should be of len > 1.")
		# Check for type of configuration
		if (keys[0] == self.confAugMultiple):
			return True
		else:
			return False

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