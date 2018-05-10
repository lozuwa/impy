"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: A class that contains specific assertion methods.
"""
import os
import sys
import numpy as np

class AssertDataTypes(object):
	def __init__(self):
		super(AssertDataTypes, self).__init__()
		
	def assertNumpyType(self, data):
		"""
		Asserts a data type as a numpy type.
		Args:
			data: A variable.
		Returns:
			A boolean. If True, then data is of numpy type.
			If false, then data is not of numpy type.
		"""
		if (type(data) == np.ndarray):
			return True
		else:
			return False

def assertNumpyType(data = None):
	"""
	Asserts a data type as a numpy type.
	Args:
		data: A variable.
	Returns:
		A boolean. If True, then data is of numpy type.
		If false, then data is not of numpy type.
	"""
	if (type(data) == np.ndarray):
		return True
	else:
		return False