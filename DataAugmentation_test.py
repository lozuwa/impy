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
from DataAugmentation import *

class DataAugmentation_test(unittest.TestCase):
	
	def setUp(self):
		self.frame = cv2.imread("cv.jpg")
		self.bndbox = [100, 100, 150, 150]
		self.da = DataAugmentation()

	def tearDown(self):
		pass

	def test_horizontal_flip(self):
		frame = self.da.horizontalFlip(self.frame)
		cv2.imshow("__frameOther__", frame)
		cv2.waitKey(100)
		cv2.destroyAllWindows()

	def test_vertical_flip(self):
		frame = self.da.verticalFlip(self.frame)
		cv2.imshow("__frameOther__", frame)
		cv2.waitKey(100)
		cv2.destroyAllWindows()

	def test_jitter_boxes(self):
		frame = self.da.jitterBoxes(self.frame)
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
			print(iy, y, ix, x)
			cv2.imshow("__frameOther__", frame[iy:y, ix:x, :])
			cv2.waitKey(100)
			cv2.destroyAllWindows()
			theta += 0.5

	def test_rotation_equations(self):
		x, y, theta = 10, 10, math.pi / 2
		x, y = DataAugmentation.rotation_equations(x, y, theta)
		self.assertEqual(x, -10)
		self.assertEqual(y, 10)

	def test_center_crop(self):
		rows, cols, depth = self.frame.shape
		print("Image size: ", rows, cols, depth)
		ROIxmin, ROIymin, ROIxmax, ROIymax,\
			xmin, xmax, ymin, ymax = self.da.centerCrop(frameHeight = rows,
																									frameWidth = cols,
																									bndbxCoordinates = self.bndbox,
																									offset = 200)
		print("Center crop image: ", ROIxmin, ROIxmax, ROIymin, ROIymax)
		cropped = self.frame[ROIxmin:ROIxmax, ROIymin:ROIymax, :]
		print("Center cropped: ", cropped.shape)
		cv2.imshow("__center_crop__", cropped)
		cv2.waitKey(3000)
		cv2.destroyAllWindows()

	def test_crop_with_translation(self):
		rows, cols, depth = self.frame.shape
		ROIxmin, ROIxmax, ROIymin, ROIymax,\
			xmin, xmax, ymin, ymax = self.da.cropWithTranslation(frameHeight = rows,
																									frameWidth = cols,
																									bndbxCoordinates = self.bndbox,
																									offset = 100)
		cropped = self.frame[ROIxmin:ROIxmax, ROIymin:ROIymax, :]
		cv2.imshow("__crop_translation__", cropped)
		cv2.waitKey(1000)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	unittest.main()
