import unittest
import numpy as np
import cv2
from ImagePreprocess import *

class ImageProcess_test(unittest.TestCase):

	def setUp(self):
		self.prep = ImagePreprocess()

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

	def test_adjustImage(self):
		# Local variables
		frameHeight = 3096
		frameWidth = 4128
		offset = 1032
		# Testing bounding boxes
		# bndboxes = [[1487, 1728, 1602, 1832], [2406, 1814, 2521, 1943], \
		bndboxes = [[3723, 1461, 3832, 1547]]
		# print("Bounding boxes before: ", bndboxes)
		# print("Cropping box: ", min_x, min_y, max_x, max_y)
		RoiXMin, RoiYMin, RoiXMax,\
		RoiYMax = self.prep.adjustImage(frameHeight = frameHeight,
																					frameWidth = frameWidth,
																					boundingBoxes = bndboxes,
																					offset = offset)
		# print("Cropping coordinates: ", RoiXMin, RoiYMin, RoiXMax, RoiYMax)
		# print("Size after cropping: ", (RoiXMax-RoiXMin), (RoiYMax-RoiYMin))
		# print("Bounding boxes after: ", bndboxes)
		# Assertions
		self.assertGreaterEqual((RoiXMax-RoiXMin), offset-100, "Cropping frame is " + \
																														"much smaller than offset.")
		self.assertGreaterEqual((RoiYMax-RoiYMin), offset-100, "Cropping frame is " + \
																														"much smaller than offset.")
		# for bdx in bndboxes:
		#   self.assertGreaterEqual(bdx[0], 0, "Xmin is negative")
		#   self.assertGreaterEqual(bdx[1], 0, "Ymin is negative")
		#   self.assertLessEqual(bdx[2], frameWidth, "Xmax is negative")
		#   self.assertLessEqual(bdx[3], frameHeight, "Ymax is negative")

if __name__ == "__main__":
	unittest.main()
