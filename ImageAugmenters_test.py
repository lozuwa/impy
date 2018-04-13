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
from ImageAugmenters import *

class ImageAugmenters_test(unittest.TestCase):

	def setUp(self):
		# 281x281x3
		self.frame = cv2.imread("cv.jpg")
		self.bndbox = [100, 100, 150, 150]
		self.da = ImageAugmenters()

	def tearDown(self):
		pass

	def test_jitter_boxes(self):
		frame = self.da.jitterBoxes(frame = self.frame,
																size = (10,10))
		cv2.imshow("__frameOther__", frame)
		cv2.waitKey(100)
		cv2.destroyAllWindows()

	def test_random_rotation(self):
		theta = 0
		for i in range(1):
			frame, ps = self.da.randomRotation(self.frame, 
																				self.bndbox,
																				theta)
			ix, iy, x, y = ps
			# print(ps)
			cv2.imshow("__frame__", frame)
			# print(iy, y, ix, x)
			cv2.imshow("__frameOther__", frame[iy:y, ix:x, :])
			cv2.waitKey(100)
			cv2.destroyAllWindows()
			theta += 0.5

	def test_fancyPCA(self):
		frame_pca = self.da.fancyPCA(frame = self.frame)
		# cv2.imwrite("test.jpg", frame_pca)
		cv2.imshow("__original__", self.frame)
		cv2.imshow("__pca__", frame_pca)
		cv2.waitKey(100)
		cv2.destroyAllWindows()

	def test_rotation_equations(self):
		x, y, theta = 10, 10, math.pi / 2
		x, y = ImageAugmenters.rotation_equations(x, y, theta)
		self.assertEqual(x, -10)
		self.assertEqual(y, 10)

if __name__ == "__main__":
	unittest.main()
