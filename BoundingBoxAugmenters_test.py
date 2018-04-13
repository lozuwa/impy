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
from BoundingBoxAugmenters import *

class BoundingBoxAugmenters_test(unittest.TestCase):
	
	def setUp(self):
		# 281x281x3
		self.frame = cv2.imread("cv.jpg")
		self.bndbox = [100, 100, 150, 150]
		self.augmenter = BoundingBoxAugmenters()

	def tearDown(self):
		pass

	def test_scale(self):
		# Prepare data
		frame = np.zeros([200, 200, 3])
		boundingBoxes = [[100, 100, 150, 150]]
		# Reduce test
		resizeSize = (100, 100)
		scaled_frame, bndboxes = self.augmenter.scale(frame = frame,
																									boundingBoxes = boundingBoxes,
																									resizeSize = resizeSize,
																									interpolationMethod = 2)
		self.assertEqual(scaled_frame.shape[0], 100)
		self.assertEqual(scaled_frame.shape[1], 100)
		self.assertAlmostEqual(bndboxes[0][0], 50)
		self.assertAlmostEqual(bndboxes[0][1], 50)
		self.assertAlmostEqual(bndboxes[0][2], 75)
		self.assertAlmostEqual(bndboxes[0][3], 75)
		# Increase test
		boundingBoxes = [[100, 100, 150, 150]]
		resizeSize = (500, 500)
		scaled_frame, bndboxes = self.augmenter.scale(frame = frame,
																									boundingBoxes = boundingBoxes,
																									resizeSize = resizeSize,
																									interpolationMethod = 2)
		self.assertEqual(scaled_frame.shape[0], 500)
		self.assertEqual(scaled_frame.shape[1], 500)
		# Assert coordinates are equal for at least 10 units
		self.assertAlmostEqual(np.abs(bndboxes[0][0] - 250)//10, 0)
		self.assertAlmostEqual(np.abs(bndboxes[0][1] - 250)//10, 0)
		self.assertAlmostEqual(np.abs(bndboxes[0][2] - 375)//10, 0)
		self.assertAlmostEqual(np.abs(bndboxes[0][3] - 375)//10, 0)
		# Stretch test
		boundingBoxes = [[100, 100, 150, 150]]
		resizeSize = (400, 200)
		scaled_frame, bndboxes = self.augmenter.scale(frame = frame,
																									boundingBoxes = boundingBoxes,
																									resizeSize = resizeSize,
																									interpolationMethod = 2)
		self.assertEqual(scaled_frame.shape[0], 200)
		self.assertEqual(scaled_frame.shape[1], 400)
		self.assertAlmostEqual(bndboxes[0][0], 200)
		self.assertAlmostEqual(bndboxes[0][1], 100)
		self.assertAlmostEqual(bndboxes[0][2], 300)
		self.assertAlmostEqual(bndboxes[0][3], 150)

	def test_random_crop(self):
		# Prepare data
		boundingBoxes = [[100, 100, 150, 150]]
		size = (25, 25, 3)
		# Apply transformation
		boundingBoxes = self.augmenter.randomCrop(boundingBoxes = boundingBoxes,
																							size = size)
		# print(boundingBoxes)
		# Assert values
		for i in range(len(boundingBoxes)):
			ix, iy, x, y = boundingBoxes[i]
			# print(ix, iy, x, y)
			# Assert sizes
			self.assertEqual((x-ix), size[0])
			self.assertEqual((y-iy), size[1])
			# Assert coordinates' positions
			self.assertGreaterEqual(ix, boundingBoxes[0][0])
			self.assertGreaterEqual(iy, boundingBoxes[0][1])
			self.assertLessEqual(x, boundingBoxes[0][2])
			self.assertLessEqual(y, boundingBoxes[0][3])

	def test_random_pad(self):
		# Prepare data
		boundingBoxes = [[100, 100, 150, 150]]
		size = (50, 50)
		# Apply transformation
		boundingBoxes = self.augmenter.randomPad(boundingBoxes = boundingBoxes,
																							frameHeight = 200,
																							frameWidth = 200,
																							size = size)
		# print(boundingBoxes)
		self.assertLessEqual(boundingBoxes[0][0], 100)
		self.assertLessEqual(boundingBoxes[0][1], 100)
		self.assertGreaterEqual(boundingBoxes[0][2], 150)
		self.assertGreaterEqual(boundingBoxes[0][3], 150)
	
	def test_jitter_boxes(self):
		boundingBoxes = [[100, 100, 150, 150], [100,150,150,200]]
		frame = self.augmenter.jitterBoxes(frame = self.frame,
																				boundingBoxes = boundingBoxes,
																				size = (10,10),
																				quantity = 2)
		# cv2.imshow("__jitterBoxes__", frame)
		# cv2.waitKey(5000)
		# cv2.destroyAllWindows()

	def test_horizontal_flip(self):
		# Prepare data.
		bndBxs0 = [[70,100,120,150], [150,100,200,150]]
		# Perform horizontal flip.
		frame = self.augmenter.horizontalFlip(frame = self.frame.copy(),
																					boundingBoxes = bndBxs0)
		# Visualization.
		# cv2.imshow("__original__", self.frame)
		# cv2.imshow("__horizontalFlip__", frame)
		# cv2.waitKey(5000)
		# cv2.destroyAllWindows()

	def test_vertical_flip(self):
		# Prepare data.
		bndBxs0 = [[70,100,120,150], [150,100,200,150]]
		# Perform horizontal flip.
		frame = self.augmenter.verticalFlip(frame = self.frame.copy(),
																					boundingBoxes = bndBxs0)
		# Visualization.
		# cv2.imshow("__original__", self.frame)
		# cv2.imshow("__verticalFlip__", frame)
		# cv2.waitKey(5000)
		# cv2.destroyAllWindows()

	def test_dropout(self):
		boundingBoxes = [[70,100,120,150], [150,100,200,150]]
		frame = self.augmenter.dropout(frame = self.frame,
											              boundingBoxes = boundingBoxes,
											              size = (25,25),
											              threshold = 0.5)
		# cv2.imshow("__dropout__", frame)
		# cv2.waitKey(3000)
		# cv2.destroyAllWindows()

	def test_random_rotation(self):
		theta = 0
		boundingBoxes = [[70,100,120,150], [150,100,200,150]]
		# for i in range(25):
		# 	frame = self.augmenter.randomRotation(self.frame, 
		# 																				boundingBoxes,
		# 																				theta)
		# 	theta += 0.3
		# 	cv2.imshow("__frame__", frame)
		# 	cv2.waitKey(250)

	def test_rotation_equations(self):
		x, y, theta = 10, 10, math.pi / 2
		x, y = BoundingBoxAugmenters.rotation_equations(x, y, theta)
		self.assertEqual(x, -10)
		self.assertEqual(y, 10)

if __name__ == "__main__":
	unittest.main()
