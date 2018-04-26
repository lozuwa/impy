import os
import json

try:
	from .SupportedDataAugmentationConfigurations import *
except:
	from SupportedDataAugmentationConfigurations import *

confs = SupportedDataAugmentationConfigurations()

class AssertJsonConfiguration():
	def __init__(self, file = None):
		super(AssertJsonConfiguration, self).__init__()
		# Assertions
		if (file == None):
			raise ValueError("ERROR: File parameter cannot be None.")
		if (not (os.path.isfile(file))):
			raise Exception("ERROR: Path to {} does not exist.".format(file))
		# Class variables
		f = open(file)
		self.file = json.load(f)
		f.close()

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
		if (augmentation in confs.methodsBndbxs):
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
		if (augmentation in confs.methodsColor):
			return True
		else:
			return False

	def runAllAssertions(self):
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
		if (keys[0] in confs.dataAugTypes):
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
		if (keys[0] == confs.confBndbxs):
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
		if (keys[0] == confs.confGeometric):
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
		if (keys[0] == confs.confColor):
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
		if (keys[0] == confs.confMultiple):
			return True
		else:
			return False
