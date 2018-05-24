"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: This class implements methods to deal with the 
configuration files.
"""
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
		self.supportedDataAugmentationTypes = [self.confAugMultiple, self.confAugBndbxs, \
																				self.confAugGeometric, self.confAugColor]
		# Augmenter methods for bounding boxes.
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
		# Augmenter methods for color space.
		self.invertColor = "invertColor"
		self.histogramEqualization = "histogramEqualization"
		self.changeBrightness = "changeBrightness"
		self.sharpening = "sharpening"
		self.addGaussianNoise = "addGaussianNoise"
		self.gaussianBlur = "gaussianBlur"
		self.averageBlur = "averageBlur"
		self.medianBlur = "medianBlur"
		self.bilateralBlur = "bilateralBlur"
		self.shiftColors = "shiftColors"
		self.fancyPCA = "fancyPCA"
		self.colorMethods = [self.invertColor, self.histogramEqualization, self.changeBrightness, \
													self.sharpening, self.addGaussianNoise, self.gaussianBlur, \
													self.averageBlur, self.medianBlur, self.bilateralBlur, \
													self.shiftColors, self.fancyPCA]
		# Geometric augmenters.
		self.translate = "translate"
		self.geometricMethods = [self.scale, self.crop, self.translate, self.jitterBoxes, \
														self.horizontalFlip, self.verticalFlip, self.rotation, self.dropout]


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

	def isValidGeometricAugmentation(self, augmentation = None):
		"""
		Assertas that augmentation is a valid geometric augmentation supported by the library.
		Args:
			augmentation: A string that contains an augmentation type.
		Returns:
			A boolean that if True means the augmentation type is vlaid, 
			otherwise it is false.
		"""
		# Assertions
		if (augmentation == None):
			raise ValueError("ERROR: augmentation parameter cannot be empty." +\
											" Report this problem.")
		# Logic
		if (augmentation in self.geometricMethods):
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
			self.isMultipleConfFileValid()
			return 3
		else:
			raise Exception("The configuration is not valid: {}.".format(keys) +\
							"bndbx: {} geometric: {} color: {}".format(rBndbx, rGeometric, rColor))

	def isMultipleConfFileValid(self):
		# Assert sequential follows multiple_image_augmentations.
		if (not ("Sequential" in self.file["multiple_image_augmentations"])):
			raise ValueError("Data after multiple_image_augmentations is not recognized." +\
											" You should place Sequential.")
		# Multiple augmentation configurations, get a list of hash maps of all the confs.
		listAugmentersConfs = self.file["multiple_image_augmentations"]["Sequential"]
		# Assert listAugmentersConfs is of type list.
		if (not (type(listAugmentersConfs) == list)):
			raise TyperError("The data inside multiple_image_augmentations/Sequential" +\
												" should be a list.")
		for i in range(len(listAugmentersConfs)):
			augmentationConf = list(listAugmentersConfs[i].keys())[0]
			# Assert the configuration is valid.
			if (not (self.isBndBxAugConfFile(keys = [augmentationConf]) or
					self.isColorConfFile(keys = [augmentationConf]))):
				raise Exception("{} is not a valid configuration for multiple_image_augmentations."\
													.format(augmentationConf))
			# Assert the data inside the configuration is a list.
			if (not ("Sequential" in listAugmentersConfs[i][augmentationConf])):
				raise ValueError("Data after multiple_image_augmentations/Sequential/{}"\
												.format(augmentationConf) + " should be Sequential.")
			listAugmentersConfsTypes = listAugmentersConfs[i][augmentationConf]["Sequential"]
			# Assert listAugmentersConfs is of type list.
			if (not (type(listAugmentersConfsTypes) == list)):
				raise TyperError("The data inside multiple_image_augmentations/Sequential" +\
													"/{}/Sequential should be a list.".format(augmentationConf))
			for j in range(len(listAugmentersConfsTypes)):
				# Get augmentation type and its parameters.
				augmentationInConfType = list(listAugmentersConfsTypes[j].keys())[0]
				parameters = listAugmentersConfsTypes[j][augmentationInConfType]
				if (augmentationConf == "bounding_box_augmenters"):
					# Validate the type of augmentation.
					if (not self.isValidBoundingBoxAugmentation(augmentation = augmentationInConfType)):
						raise ValueError("Type of configuration for {}/{} not valid. Use a corresponding augmenter."\
														.format(augmentationConf, augmentationInConfType))
					# Validate the content of the augmentation.
					self.validateBoundingBoxAugmentation(augmentationType = augmentationInConfType, \
																									parameters = parameters)
				elif (augmentationConf == "image_color_augmenters"):
					# Validate the type of augmentation.
					if (self.isValidBoundingBoxAugmentation(augmentation = augmentationInConfType)):
						raise ValueError("Type of configuration for {}/{} not valid. Use a corresponding augmenter."\
														.format(augmentationConf, augmentationInConfType))
					# Validate the content of the augmentation.
					self.validateColorAugmentation(augmentationType = augmentationInConfType,\
																					parameters = parameters)

	def isValidConfFile(self, keys = None):
		# Check len of keys.
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

	def validateBoundingBoxAugmentation(self, augmentationType = None, parameters = None):
		"""
		Applies a bounding box augmentation making sure all the parameters exist or are 
		correct.
		Args:
			augmentationType: A string that contains a type of augmentation.
			parameters: A hashmap that contains parameters for the respective type 
								of augmentation.
		Returns:
			A tensor that contains a frame with the respective transformation.
		"""
		# Logic.
		if (augmentationType == "scale"):
			# Apply scaling.
			if (not ("size" in parameters)):
				raise Exception("Scale requires parameter size.")
			if (not ("zoom" in parameters)):
				parameters["zoom"] = None
			if (not ("interpolationMethod" in parameters)):
				parameters["interpolationMethod"] = None
		elif (augmentationType == "crop"):
			# Apply crop.
			if (not ("size" in parameters)):
				parameters["size"] = None
		elif (augmentationType == "pad"):
			# Apply pad.
			if (not ("size" in parameters)):
				raise Exception("Pad requires parameter size.")
		elif (augmentationType == "jitterBoxes"):
			# Apply jitter boxes.
			if (not ("size" in parameters)):
				raise Exception("JitterBoxes requires parameter size.")
			if (not ("quantity" in parameters)):
				parameters["quantity"] = None
		elif (augmentationType == "horizontalFlip"):
			pass
		elif (augmentationType == "verticalFlip"):
			pass
		elif (augmentationType == "rotation"):
			# Apply rotation.
			if (not ("theta" in parameters)):
				theta = None
				#raise Exception("ERROR: Rotation requires parameter theta.")
			else:
				theta = parameters["theta"]
		elif (augmentationType == "dropout"):
			# Apply dropout.
			if (not ("size" in parameters)):
				raise Exception("Dropout requires parameter size.")
			if (not ("threshold" in parameters)):
				parameters["threshold"] = None
		else:
			raise Exception("Bounding box augmentation type not supported: {}."\
											.format(augmentationType))

	def validateColorAugmentation(self, augmentationType = None, parameters = None):
		"""
		Applies a color augmentation making sure all the parameters exist or are 
		correct.
		Args:
			augmentationType: A string that contains a type of augmentation.
			parameters: A hashmap that contains parameters for the respective type 
								of augmentation.
		Returns:
			A tensor that contains a frame with the respective transformation.
		"""
		# Logic.
		if (augmentationType == "invertColor"):
			if (not ("CSpace" in parameters)):
				parameters["CSpace"] = None
		elif (augmentationType == "histogramEqualization"):
			if (not ("equalizationType" in parameters)):
				parameters["equalizationType"] = None
		elif (augmentationType == "changeBrightness"):
			if (not ("coefficient" in parameters)):
				raise AttributeError("coefficient for changeBrightness must be specified.")
		elif (augmentationType == "sharpening"):
			if (not ("weight" in parameters)):
				parameters["weight"] = None
		elif (augmentationType == "addGaussianNoise"):
			if (not ("coefficient" in parameters)):
				parameters["coefficient"] = None
		elif (augmentationType == "gaussianBlur"):
			if (not ("sigma" in parameters)):
				parameters["sigma"] = None
			if (not ("kernelSize" in parameters)):
				parameters["kernelSize"] = None
		elif (augmentationType == "averageBlur"):
			if (not ("kernelSize" in parameters)):
				parameters["kernelSize"] = None
		elif (augmentationType == "medianBlur"):
			if (not ("coefficient" in parameters)):
				parameters["coefficient"] = None
		elif (augmentationType == "bilateralBlur"):
			if (not ("d" in parameters)):
				parameters["d"] = None
			if (not ("sigmaColor" in parameters)):
				parameters["sigmaColor"] = None
			if (not ("sigmaSpace" in parameters)):
				parameters["sigmaSpace"] = None
		elif (augmentationType == "shiftColors"):
			pass
		elif (augmentationType == "fancyPCA"):
			pass
		else:
			raise Exception("Color augmentation type not supported: {}."\
											.format(augmentationType))

	def validateGeometricAugmentation(self, augmentationType = None, parameters = None):
		"""
		Applies a geometric augmentation making sure all the parameters exist or are 
		correct.
		Args:
			augmentationType: A string that contains a type of augmentation.
			parameters: A hashmap that contains parameters for the respective type 
								of augmentation.
		Returns:
			A tensor that contains a frame with the respective transformation.
		"""
		# Logic
		if (augmentationType == "scale"):
			# Apply scaling
			if (not ("size" in parameters)):
				raise Exception("ERROR: Scale requires parameter size.")
			if (not ("interpolationMethod" in parameters)):
				print("WARNING: Interpolation method for scale will be set to default value.")
		elif (augmentationType == "crop"):
			# Apply crop
			if (not ("size" in parameters)):
				parameters["size"] = None
				print("WARNING: Size for crop will be set to default value.")
		elif (augmentationType == "translate"):
			# Apply pad
			if (not ("offset" in parameters)):
				raise Exception("Pad requires parameter offset.")
		elif (augmentationType == "jitterBoxes"):
			# Apply jitter boxes
			if (not ("size" in parameters)):
				raise Exception("JitterBoxes requires parameter size.")
			if (not ("quantity" in parameters)):
				parameters["quantity"] = 10
				print("WARNING: Quantity for jitter boxes will be set to its default value.")
			if (not ("color" in parameters)):
				parameters["color"] = [255,255,255]
				print("WARNING: Color for jitter boxes will be set to its default value.")
		elif (augmentationType == "horizontalFlip"):
			pass
		elif (augmentationType == "verticalFlip"):
			pass
		elif (augmentationType == "rotation"):
			# Apply rotation
			if (not ("theta" in parameters)):
				pass
