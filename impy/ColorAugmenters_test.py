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
		img_path = os.path.join(os.getcwd(), "../../cars_dataset/images/cars0.png")
		assert os.path.isfile(img_path)
		# Image
		self.frame = cv2.imread(img_path)
		# Augmenters
		self.augmenter = ColorAugmenters()
		# Testing options
		self.visualize = True
		self.waitTime = 50
		self.windowSize = (800, 800)

	def tearDown(self):
		pass

	# def test_invert_color(self):
	# 	if (self.visualize):
	# 		frame = self.augmenter.invertColor(frame = self.frame,
	# 																				CSpace = [True, True, True])
	# 		cv2.namedWindow("__invertColor__", 0)
	# 		# cv2.resizeWindow("__invertColor__", self.windowSize)
	# 		cv2.imshow("__invertColor__", frame)
	# 		cv2.waitKey(self.waitTime)
	# 		cv2.destroyAllWindows()

	# def test_histogram_equalization(self):
	# 	if (self.visualize):
	# 		# Type 0
	# 		frame = self.augmenter.histogramEqualization(frame = self.frame,
	# 																								equalizationType = 0)
	# 		cv2.namedWindow("__equalizationType__", 0)
	# 		# cv2.resizeWindow("__equalizationType__", self.windowSize);
	# 		cv2.imshow("__equalizationType__", frame)
	# 		cv2.waitKey(self.waitTime)
	# 		cv2.destroyAllWindows()
	# 		self.assertEqual(self.frame.shape[0], frame.shape[0])
	# 		self.assertEqual(self.frame.shape[1], frame.shape[1])
	# 		self.assertEqual(self.frame.dtype, frame.dtype)
	# 		# Type 1
	# 		frame = self.augmenter.histogramEqualization(frame = self.frame,
	# 																								equalizationType = 1)
	# 		cv2.namedWindow("__equalizationType__", 1)
	# 		# cv2.resizeWindow("__equalizationType__", self.windowSize);
	# 		cv2.imshow("__equalizationType__", frame)
	# 		cv2.waitKey(self.waitTime)
	# 		cv2.destroyAllWindows()
	# 		self.assertEqual(self.frame.shape[0], frame.shape[0])
	# 		self.assertEqual(self.frame.shape[1], frame.shape[1])
	# 		self.assertEqual(self.frame.dtype, frame.dtype)

	# def test_change_brightness(self):
	# 	if (self.visualize):
	# 		c = 0.4
	# 		for i in range(10):
	# 			frame = self.augmenter.changeBrightness(frame = self.frame,
	# 																							coefficient = c)
	# 			cv2.namedWindow("__changeBrightness__", 0)
	# 			# cv2.resizeWindow("__changeBrightness__", self.windowSize);
	# 			cv2.imshow("__changeBrightness__", frame)
	# 			cv2.waitKey(self.waitTime)
	# 			cv2.destroyAllWindows()
	# 			self.assertEqual(self.frame.shape[0], frame.shape[0])
	# 			self.assertEqual(self.frame.shape[1], frame.shape[1])
	# 			self.assertEqual(self.frame.dtype, frame.dtype)
	# 			c += 0.2

	# def test_sharpening(self):
	# 	if (self.visualize):
	# 		w = 0.2
	# 		for i in range(10):
	# 			frame = self.augmenter.sharpening(frame = self.frame,
	# 																					weight = w)
	# 			cv2.namedWindow("__sharpening__", 0)
	# 			# cv2.resizeWindow("__sharpening__", self.windowSize);
	# 			cv2.imshow("__sharpening__", frame)
	# 			cv2.waitKey(self.waitTime)
	# 			cv2.destroyAllWindows()
	# 			self.assertEqual(self.frame.shape[0], frame.shape[0])
	# 			self.assertEqual(self.frame.shape[1], frame.shape[1])
	# 			self.assertEqual(self.frame.dtype, frame.dtype)
	# 			w += 0.2

	# def test_add_gaussian_noise(self):
	# 	if (self.visualize):
	# 		frame = self.augmenter.addGaussianNoise(frame = self.frame,
	# 																						coefficient = 0.5)
	# 		cv2.namedWindow("__addGaussianNoise__", 0)
	# 		# cv2.resizeWindow("__addGaussianNoise__", self.windowSize);
	# 		cv2.imshow("__addGaussianNoise__", frame)
	# 		cv2.waitKey(self.waitTime)
	# 		cv2.destroyAllWindows()

	# def test_gaussian_blur(self):
	# 	if (self.visualize):
	# 		vals = [5,7,9,10,15]
	# 		for each in vals:
	# 			print(each)
	# 			frame = self.augmenter.gaussianBlur(frame = self.frame,
	# 																					sigma = each)
	# 			cv2.namedWindow("__gaussianBlur__", 0)
	# 			cv2.imshow("__gaussianBlur__", frame)
	# 			cv2.waitKey(self.waitTime)
	# 			cv2.destroyAllWindows()

	def test_average_blur(self):
		if (self.visualize):
			for i in range(5):
				frame = self.augmenter.averageBlur(frame = self.frame,
																	kernelSize = [i+2, i+2])
				cv2.namedWindow("__averageBlur__", 0)
				cv2.imshow("__averageBlur__", frame)
				cv2.waitKey(self.waitTime)
				cv2.destroyAllWindows()
				self.assertEqual(self.frame.shape[0], frame.shape[0])
				self.assertEqual(self.frame.shape[1], frame.shape[1])
				self.assertEqual(self.frame.dtype, frame.dtype)

	# def test_median_blur(self):
	# 	if (self.visualize):
	# 		vals = [1,3,5,7,9]
	# 		for each in vals:
	# 			frame = self.augmenter.medianBlur(frame = self.frame,
	# 																				coefficient = each)
	# 			cv2.namedWindow("__medianBlur__", 0)
	# 			cv2.imshow("__medianBlur__", frame)
	# 			cv2.waitKey(self.waitTime)
	# 			cv2.destroyAllWindows()
	# 			self.assertEqual(self.frame.shape[0], frame.shape[0])
	# 			self.assertEqual(self.frame.shape[1], frame.shape[1])
	# 			self.assertEqual(self.frame.dtype, frame.dtype)

	# def test_bilateral_blur(self):
	# 	if (self.visualize):
	# 		ds = [1,3,5,7,9]
	# 		scs = []
	# 		sss = []
	# 		for d, sc, ss in zip(ds, scs, sss):
	# 			frame = self.augmenter.bilateralBlur(frame = self.frame,
	# 																				d = d,
	# 																				sigmaColor = sc,
	# 																				sigmaSpace = ss)
	# 			cv2.namedWindow("__bilateralBlur__", 0)
	# 			cv2.imshow("__bilateralBlur__", frame)
	# 			cv2.waitKey(self.waitTime)
	# 			cv2.destroyAllWindows()
	# 			self.assertEqual(self.frame.shape[0], frame.shape[0])
	# 			self.assertEqual(self.frame.shape[1], frame.shape[1])
	# 			self.assertEqual(self.frame.dtype, frame.dtype)

	# def test_shift_colors(self):
	# 	if (self.visualize):
	# 		frame = self.augmenter.shiftColors(frame = self.frame)
	# 		cv2.namedWindow("__shiftColors__", 0)
	# 		# cv2.resizeWindow("__shiftColors__", self.windowSize);
	# 		cv2.imshow("__shiftColors__", frame)
	# 		cv2.waitKey(self.waitTime)
	# 		cv2.destroyAllWindows()
	# 		self.assertEqual(self.frame.shape[0], frame.shape[0])
	# 		self.assertEqual(self.frame.shape[1], frame.shape[1])
	# 		self.assertEqual(self.frame.dtype, frame.dtype)

	# def test_fancyPCA(self):
	# 	if (self.visualize):
	# 		frame = self.augmenter.fancyPCA(frame = self.frame)
	# 		# cv2.imwrite("/home/pfm/Downloads/fff.jpg", frame)
	# 		cv2.namedWindow("__fancyPCA__", 0)
	# 		# cv2.resizeWindow("__fancyPCA__", self.windowSize);
	# 		cv2.imshow("__fancyPCA__", frame)
	# 		cv2.waitKey(self.waitTime)
	# 		cv2.destroyAllWindows()

if __name__ == "__main__":
	unittest.main()
