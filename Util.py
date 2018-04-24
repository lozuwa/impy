"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Implements common file, logic 
operations.
"""
import os
import datetime
import re
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
			raise ValueError("ERROR: Folder name parameter cannot be emtpy.")
		if (os.path.isdir(folder_name)):
			print("INFO: Folder already exists: {}".format(os.path.split(folder_name)[1]))
		else:
			try:
				os.mkdir(folder_name)
			except:
				raise Exception("ERROR: Folder {} could not be created.".format(folder_name))

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
	def detect_file_extension(filename):
		"""
		Detect the extension of a file.
		Args:
			filename: A string that contains the name of a file.
		Returns:
			A string that contains the extension of the file. It 
			returns None if there is no extension.
		"""
		# Assertions
		if (filename == None):
			raise ValueError("Filename cannot be empty.")
		# Logic
		if (filename.endswith(".jpg")):
			return ".jpg"
		elif (filename.endswith(".png")):
			return ".png"
		else:
			return None

	@staticmethod
	def assert_file_extension(filename = None, extension = None):
		"""
		Assert the extension of a file.
		Args:
			filename: A string that contains the name of a file.
			extension: A string that contains a file extension.
		Returns:
			A boolean that if True means the extension has been verified. 
			Otherwise, the extension is not asserted.
		"""
		# Assertions
		if (filename == None):
			raise ValueError("Parameter filename cannot be empty.")
		if (type(filename) == str):
			pass
		else:
			raise TypeError("Parameter filename has to be string: {}".format(type(filename)))
		if (extension == None):
			raise ValueError("Parameter extension cannot be empty.")
		if (type(extension) == str):
			pass
		else:
			raise TypeError("Parameter extension has to be string: {}".format(type(extension))) 
		# Logic
		if ()


