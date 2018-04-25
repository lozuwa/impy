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
		img_path = os.path.join(os.getcwd(), "tests/cars_dataset/images/cars0.png")
		assert os.path.isfile(img_path)
		# Image
		self.frame = cv2.imread(img_path)
		# Augmenters
		self.augmenter = GeometricAugmenters()
		# Testing options
		self.visualize = True
		self.waitTime = 2000
		self.windowSize = (800, 800)

	def tearDown(self):
		pass

	def test_scale(self):
		if (self.visualize):
			frame = self.augmenter.scale(frame = self.frame,
																	size = (100, 100),
																	interpolationMethod = 1)
			cv2.namedWindow("__scale__", 0)
			cv2.resizeWindow("__scale__", self.windowSize);
			cv2.imshow("__scale__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_translate(self):
		if (self.visualize):
			frame = self.augmenter.translate(frame = self.frame,
																				offset = (100, 100))
			cv2.namedWindow("__translate__", 0)
			cv2.resizeWindow("__translate__", self.windowSize);
			cv2.imshow("__translate__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_jitter_boxes(self):
		if (self.visualize):
			frame = self.augmenter.jitterBoxes(frame = self.frame,
																				size = (10,10),
																				quantity = 30)
			cv2.namedWindow("__jitterBoxes__", 0)
			cv2.resizeWindow("__jitterBoxes__", self.windowSize);
			cv2.imshow("__jitterBoxes__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_horizontal_flip(self):
		if (self.visualize):
			frame = self.augmenter.horizontalFlip(frame = self.frame)
			cv2.namedWindow("__horizontalFlip__", 0)
			cv2.resizeWindow("__horizontalFlip__", self.windowSize);
			cv2.imshow("__horizontalFlip__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_vertical_flip(self):
		if (self.visualize):
			frame = self.augmenter.verticalFlip(frame = self.frame)
			cv2.namedWindow("__verticalFlip__", 0)
			cv2.resizeWindow("__verticalFlip__", self.windowSize);
			cv2.imshow("__verticalFlip__", frame)
			cv2.waitKey(self.waitTime)
			cv2.destroyAllWindows()

	def test_rotation(self):
		if (self.visualize):
			theta = 0
			height, width = self.frame.shape[0], self.frame.shape[1]
			for i in range(3):
				frame, ps = self.augmenter.rotation(frame = self.frame,
																									bndbox = [0,0,width,height],
																									theta = theta)
				ix, iy, x, y = ps
				cv2.namedWindow("__rotation__", 0)
				cv2.resizeWindow("__rotation__", self.windowSize);
				cv2.imshow("__rotation__", frame[iy:y, ix:x, :])
				cv2.waitKey(2500)
				cv2.destroyAllWindows()
				theta += 0.5

if __name__ == "__main__":
	unittest.main()
