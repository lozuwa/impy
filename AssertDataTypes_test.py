"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: A class that contains specific assertion methods.
"""
import unittest
import numpy as np
from AssertDataTypes import *

class AssertDataTypes_test(unittest.TestCase):

	def setUp(self):
		self.assertion = AssertDataTypes()

	def tearDown(self):
		pass

	def test_assert_numpy_type(self):
		# Prepare data
		dummy = np.zeros([100, 100, 3])
		# Assert numpy type
		result = self.assertion.assertNumpyType(dummy)
		self.assertTrue(result)
		# Assert not numpy type
		dummy = ["abc", 2, 3.0, (4,5), [1,2,3]]
		for each in dummy:
			result = self.assertion.assertNumpyType(each)
			self.assertFalse(result)

if __name__ == "__main__":
	unittest.main()