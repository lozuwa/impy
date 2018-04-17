"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Implements common file, logic 
operations.
"""
import os
import datetime
import json
import numpy as np

class Util(object):
	def __init__(self):
		super(Util, self).__init__()

	@staticmethod
	def create_folder(folder_name = None):
		"""
		Creates a folder.
		Args:
			folder_name: A string that contains the name of the folder to be created.
		Returns:
			None
		"""
		# Assertions
		if (folder_name == None):
			raise ValueError("Folder name parameter cannot be emtpy.")
		folder_name = os.path.join(os.getcwd(), folder_name)
		if os.path.isdir(folder_name):
			print("INFO: Folder already exists :: {}".format(os.path.split(folder_name)[1]))
		else:
			try:
				os.mkdir(folder_name)
			except:
				raise Exception("Folder could not be created")

	@staticmethod
	def create_random_name(name = None,
													length = None):
		"""
		Generates a name based on the name parameter.
		Args:
			name: A string that contains the name of the image that is being labeled.
			length: An int that defines the length of the random string in the name
							to be created.
		Returns:
			A string that contains a new name with respect to the current time.
		"""
		# Assertions
		if (name == None):
			raise ValueError("Name parameter cannot be empty.")
		if (length == None):
			raise ValueError("Length parameter cannot be empty.")
		# Local variables
		abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", 
			"m", "n", "o", "p", "q","r", "s", "t", "u", "v", "w", "x", "y", "z"]
		# Get current time.
		now = datetime.datetime.now()
		# Create random 4-length string.
		var_str = ""
		for i in range(4):
			random_index = int(np.random.rand()*(len(abc)-1))
			var_str += abc[random_index]
		# Append created variables.
		new_name = "{}_{}_{}_{}_{}_{}".format(name,
																					now.year,
																					now.month,
																					now.hour,
																					now.microsecond,
																					var_str)
		# Return string
		return new_name

	@staticmethod
	def decode_json(data):
		"""
		Decode a json file.
		Args:
			data: A json file.
		Returns:
			A list that contains the
		"""
		# Assertions
		if (not os.path.isfile(augmentations)):
			raise Exception("Path to json file ({}) does not exist."\
												.format(augmentations))
		# Read json
		data = json.load(open(augmentations))
		augmentation_types = [i for i in data.keys()]
		for augmentation_type in augmentation_types:
			pass
		return augmentation_types
