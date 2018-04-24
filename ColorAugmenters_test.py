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
from ColorAugmenters import *

class ColorAugmenters_test(unittest.TestCase):

	def setUp(self):
		# Image
		self.frame = cv2.imread("tests/localization/images/cv.jpg")
		# Augmenters
		self.augmenter = ColorAugmenters()
		# Testing options
		self.visualize = True
		self.waitTime = 1000

	def tearDown(self):
		pass

	def test_invertColor(self):
		if (self.visualize):
			frame = self.augmenter.invertColor(frame = self.frame,
																					CSpace = [True, True, True])
			cv2.imshow("__invertColor__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_histogram_equalization(self):
		if (self.visualize):
			# Type 0
			frame = self.augmenter.histogramEqualization(frame = self.frame,
																									equalizationType = 0)
			cv2.imshow("__equalizationType__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()
			# Type 1
			frame = self.augmenter.histogramEqualization(frame = self.frame,
																									equalizationType = 0)
			cv2.imshow("__equalizationType__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_change_brightness(self):
		if (self.visualize):
			frame = self.augmenter.changeBrightness(frame = self.frame,
																							coefficient = 0.5)
			cv2.imshow("__changeBrightness__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_sharpening(self):
		if (self.visualize):
			frame = self.augmenter.sharpening(frame = self.frame,
																					weight = 0.3)		
			cv2.imshow("__sharpening__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_add_gaussian_noise(self):
		if (self.visualize):
			frame = self.augmenter.addGaussianNoise(frame = self.frame,
																							coefficient = 0.5)
			cv2.imshow("__addGaussianNoise__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_gaussian_blur(self):
		if (self.visualize):
			frame = self.augmenter.gaussianBlur(frame = self.frame,
																					sigma = 0.5)
			cv2.imshow("__gaussianBlur__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_shift_colors(self):
		if (self.visualize):
			frame = self.augmenter.shiftColors(frame = self.frame)
			cv2.imshow("__shiftColors__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_fancyPCA(self):
		if (self.visualize):
			frame = self.augmenter.fancyPCA(frame = self.frame)
			# cv2.imwrite("/home/pfm/Downloads/fff.jpg", frame)
			cv2.imshow("__fancyPCA__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

if __name__ == "__main__":
	unittest.main()
