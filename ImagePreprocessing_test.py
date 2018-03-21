import unittest
import numpy as np
import cv2
from ImagePreprocessing import *

class ImageProcessing_test(unittest.TestCase):

	def setUp(self):
		self.prep = ImagePreprocessing()

	def tearDown(self):
		pass

	def test_divideIntoPatchesVALIDFITALL(self):
		# Conditions
		number_patches = (3, 3)
		# Simulate image
		width = 640
		height = 480
		# Test divideIntoPatches using the VALID_FIT_ALL padding
		patches, h, w = self.prep.divideIntoPatches(imageWidth = width, 
														imageHeight = height,
														slideWindowSize = None,
														strideSize = None,
														padding = "VALID_FIT_ALL",
														numberPatches = (3,3))
		self.assertEqual(len(patches), (number_patches[0] * number_patches[1]))

	def test_divideIntoPatchesVALID(self):
		# Simulate image
		width = 640
		height = 480
		# Test divideIntoPatches using the VALID_FIT_ALL padding
		patches, h, w = self.prep.divideIntoPatches(imageWidth = width, 
														imageHeight = height,
														slideWindowSize = (100, 100),
														strideSize = (100, 100),
														padding = "VALID")
		self.assertEqual(len(patches), (24))

	def test_divideIntoPatchesSAME(self):
		# Simulate image
		width = 640
		height = 480
		# Test divideIntoPatches using the VALID_FIT_ALL padding
		patches, h, w, zh, zw = self.prep.divideIntoPatches(imageWidth = width, 
														imageHeight = height,
														slideWindowSize = (100, 100),
														strideSize = (100, 100),
														padding = "SAME")
		self.assertEqual(len(patches), (35))

if __name__ == "__main__":
	unittest.main()
