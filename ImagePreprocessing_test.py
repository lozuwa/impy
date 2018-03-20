import unittest
import numpy as np
import cv2
from ImagePreprocessing import *

class ImageProcessing_test(unittest.Testcase):

	def setUp(self):
		imPrep = ImagePreprocessing()

	def test_divideIntoPatchesVALIDFITALL(self):
		# Conditions
		number_patches = (3, 3)
		# Simulate image
		width = 640
		height = 480
		depth = 3
		frame = np.zeros([width, height, depth], np.uint8)
		# Test divideIntoPatches using the VALID_FIT_ALL padding
		patches, h, w = imPrep(width = width, 
														height = height,
														slide_window_size = None,
														stride_size = None,
														padding = "VALID_FIT_ALL",
														numberPatches = (3,3))
		assert len(patches) == (number_patches[0] * number_patches[1])

	def test_divideIntoPatchesVALID(self):
		# Simulate image
		width = 640
		height = 480
		depth = 3
		frame = np.zeros([width, height, depth], np.uint8)
		# Test divideIntoPatches using the VALID_FIT_ALL padding
		patches, h, w = imPrep(width = width, 
														height = height,
														slide_window_size = (100, 100),
														stride_size = (100, 100),
														padding = "VALID",
														numberPatches = None)
		assert len(patches) == (24)

	def test_divideIntoPatchesSAME(self):
		# Simulate image
		width = 640
		height = 480
		depth = 3
		frame = np.zeros([width, height, depth], np.uint8)
		# Test divideIntoPatches using the VALID_FIT_ALL padding
		patches, h, w = imPrep(width = width, 
														height = height,
														slide_window_size = (100, 100),
														stride_size = (100, 100),
														padding = "SAME",
														numberPatches = None)
		assert len(patches) == (35)

if __name__ == "__main__":
	unittest.main()
