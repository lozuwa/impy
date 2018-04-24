"""
package: Images2Dataset
class: DataAugmentation
Author: Rodrigo Loza
Description: Test class for data augmentation.
"""
# Libraries
import unittest
import math
import numpy as np
import cv2
from GeometricAugmenters import *

class GeometricAugmenters_test(unittest.TestCase):

	def setUp(self):
		# Image
		self.frame = cv2.imread("tests/localization/images/cv.jpg")
		# Augmenters
		self.augmenter = GeometricAugmenters()
		# Testing options
		self.visualize = True
		self.waitTime = 1000

	def tearDown(self):
		pass

	def test_scale(self):
		if (self.visualize):
			frame = self.augmenter.scale(frame = self.frame,
																	size = (100, 100),
																	interpolationMethod = 1)
			cv2.imshow("__scale__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_translate(self):
		if (self.visualize):
			frame = self.augmenter.translate(frame = self.frame,
																				offset = (100, 100))
			cv2.imshow("__translate__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_jitter_boxes(self):
		if (self.visualize):
			frame = self.augmenter.jitterBoxes(frame = self.frame,
																				size = (10,10),
																				quantity = 30)
			cv2.imshow("__jitterBoxes__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_horizontal_flip(self):
		if (self.visualize):
			frame = self.augmenter.horizontalFlip(frame = self.frame)
			cv2.imshow("__horizontalFlip__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_vertical_flip(self):
		if (self.visualize):
			frame = self.augmenter.verticalFlip(frame = self.frame)
			cv2.imshow("__verticalFlip__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_rotation(self):
		if (self.visualize):
			theta = 0
			for i in range(3):
				frame, ps = self.augmenter.rotation(frame = self.frame,
																									bndbox = [100,100,150,150],
																									theta = theta)
				ix, iy, x, y = ps
				cv2.imshow("__rotation__", frame[iy:y, ix:x, :])
				cv2.waitKey(250)
				cv2.destroyAllWindows()
				theta += 0.5

if __name__ == "__main__":
	unittest.main()
