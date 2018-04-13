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
		# 281x281x3
		self.frame = cv2.imread("cv.jpg")
		self.bndbox = [100, 100, 150, 150]
		self.da = DataAugmentation()

	def tearDown(self):
		pass

	def test_horizontal_flip(self):
		# Prepare data.
		bndBxs0 = [[70,100,120,150], [150,100,200,150]]
		frame_original = cv2.rectangle(self.frame.copy(), (bndBxs0[0][0], bndBxs0[0][1]), \
											(bndBxs0[0][2], bndBxs0[0][3]), (255, 0, 0), -1)
		frame_original = cv2.rectangle(frame_original.copy(), (bndBxs0[1][0], bndBxs0[1][1]), \
											(bndBxs0[1][2], bndBxs0[1][3]), (255, 0, 0), -1)
		# Perform vertical flip.
		frame, bndBxs1 = self.da.horizontalFlip(frame = self.frame.copy(),
																					boundingBoxes = bndBxs0)
		# Assertions.
		self.assertEqual(bndBxs0[0][0], 161)
		self.assertEqual(bndBxs0[0][2], 211)
		self.assertEqual(bndBxs0[1][0], 81)
		self.assertEqual(bndBxs0[1][2], 131)
		frame_vert_flip = cv2.rectangle(frame.copy(), (bndBxs1[0][0], bndBxs1[0][1]), \
													(bndBxs1[0][2], bndBxs1[0][3]), (0, 255, 0), -1)
		frame_vert_flip = cv2.rectangle(frame_vert_flip, (bndBxs1[1][0], bndBxs1[1][1]), \
													(bndBxs1[1][2], bndBxs1[1][3]), (0, 255, 0), -1)
		# Visualization
		# cv2.imshow("__original__", frame_original)
		# cv2.imshow("__horizontalFlip__", frame_vert_flip)
		# cv2.waitKey(5000)
		# cv2.destroyAllWindows()

	def test_vertical_flip(self):
		# Prepare data.
		bndBxs0 = [[70,100,120,150], [150,100,200,150]]
		frame_original = cv2.rectangle(self.frame.copy(), (bndBxs0[0][0], bndBxs0[0][1]), \
											(bndBxs0[0][2], bndBxs0[0][3]), (255, 0, 0), -1)
		frame_original = cv2.rectangle(frame_original.copy(), (bndBxs0[1][0], bndBxs0[1][1]), \
											(bndBxs0[1][2], bndBxs0[1][3]), (255, 0, 0), -1)
		# Perform vertical flip.
		frame, bndBxs1 = self.da.verticalFlip(frame = self.frame.copy(),
																					boundingBoxes = bndBxs0)
		# Assertions.
		self.assertEqual(bndBxs0[0][1], 131)
		self.assertEqual(bndBxs0[0][3], 181)
		self.assertEqual(bndBxs0[1][1], 131)
		self.assertEqual(bndBxs0[1][3], 181)
		frame_vert_flip = cv2.rectangle(frame.copy(), (bndBxs1[0][0], bndBxs1[0][1]), \
													(bndBxs1[0][2], bndBxs1[0][3]), (0, 255, 0), -1)
		frame_vert_flip = cv2.rectangle(frame_vert_flip, (bndBxs1[1][0], bndBxs1[1][1]), \
													(bndBxs1[1][2], bndBxs1[1][3]), (0, 255, 0), -1)
		# Visualization
		# cv2.imshow("__original__", frame_original)
		# cv2.imshow("__verticalFlip__", frame_vert_flip)
		# cv2.waitKey(1000)
		# cv2.destroyAllWindows()

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
			# print(iy, y, ix, x)
			cv2.imshow("__frameOther__", frame[iy:y, ix:x, :])
			cv2.waitKey(100)
			cv2.destroyAllWindows()
			theta += 0.5

	def test_rotation_equations(self):
		x, y, theta = 10, 10, math.pi / 2
		x, y = DataAugmentation.rotation_equations(x, y, theta)
		self.assertEqual(x, -10)
		self.assertEqual(y, 10)

	def test_crop_with_translation(self):
		rows, cols, depth = self.frame.shape
		ROIxmin, ROIxmax, ROIymin, ROIymax,\
			xmin, xmax, ymin, ymax = self.da.cropWithTranslation(frameHeight = rows,
																									frameWidth = cols,
																									bndbxCoordinates = self.bndbox,
																									offset = 100)
		cropped = self.frame[ROIxmin:ROIxmax, ROIymin:ROIymax, :]
		cv2.imshow("__crop_translation__", cropped)
		cv2.waitKey(100)
		cv2.destroyAllWindows()

	def test_fancyPCA(self):
		frame_pca = self.da.fancyPCA(frame = self.frame)
		# cv2.imwrite("test.jpg", frame_pca)
		cv2.imshow("__original__", self.frame)
		cv2.imshow("__pca__", frame_pca)
		cv2.waitKey(100)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	unittest.main()
