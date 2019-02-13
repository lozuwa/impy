"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Unit tests for the AnnotationProcessing class.
"""
import unittest
from AnnotationProcessing import *		

class AnnotationProcessing_test(unittest.TestCase):
	
	def setUp(self):
		self.proc = AnnotationProcessing()

	def tearDown(self):
		pass

	def test_iou(self):
		# Examples of boxes. 
		bndbxs0 = [[39, 63, 203, 112], [54, 66, 198, 114]]
		bndbxs1 = [[49, 75, 203, 125], [42, 78, 186, 126]]
		bndbxs2 = [[31, 69, 201, 125], [18, 63, 235, 135]]
		bndbxs3 = [[50, 72, 197, 121], [54, 72, 198, 120]]
		bndbxs4 = [[50, 50, 150, 150], [200, 200, 250, 250]]
		# Find ious.
		iou0 = self.proc.iou(bndbxs0[0], bndbxs0[1])
		iou1 = self.proc.iou(bndbxs1[0], bndbxs1[1])
		iou2 = self.proc.iou(bndbxs2[0], bndbxs2[1])
		iou3 = self.proc.iou(bndbxs3[0], bndbxs3[1])
		iou4 = self.proc.iou(bndbxs4[0], bndbxs4[1])
		# Overlapping bounding boxes.
		self.assertAlmostEqual(iou0, 0.7980, places = 3)
		self.assertAlmostEqual(iou1, 0.7899, places = 3)
		self.assertAlmostEqual(iou2, 0.6125, places = 3)
		self.assertAlmostEqual(iou3, 0.9472, places = 3)
		# No overlapping bounding boxes.
		self.assertAlmostEqual(iou4, 0.230, places = 2)

	def test_nms(self):
		# Example boxes.
		boundingBoxes0 = [(12, 84, 140, 212),
										(24, 84, 152, 212),
										(36, 84, 164, 212),
										(12, 96, 140, 224),
										(24, 96, 152, 224),
										(24, 108, 152, 236)]
		boundingBoxes1 = [(114, 60, 178, 124),
											(120, 60, 184, 124),
											(114, 66, 178, 130)]
		bndbx0 = self.proc.nonMaxSuppression(boxes = boundingBoxes0, overlapThresh = 0.6)
		# Force two bounding boxes to be correct.
		bndbx1 = self.proc.nonMaxSuppression(boxes = boundingBoxes1, overlapThresh = 0.9)
		# Assert non max suppresion algorithm.
		self.assertEqual(list(bndbx0[0]), [24, 108, 152, 236])
		self.assertEqual(list(bndbx1[0]), [114, 66, 178, 130])

if __name__ == "__main__":
	unittest.main()
