"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Testing units for ImageLocalizationDataset.
"""
import os
import unittest
from ImageLocalizationDataset import *

class ImageLocalizationDataset_test(unittest.TestCase):
	
	def setUp(self):
		self.imda = ImageLocalizationDataset(images = os.path.join(os.getcwd(), \
																									"tests", "localization", \
																									"images"),
																				annotations = os.path.join(os.getcwd(), \
																										"tests", "localization", \
																										"annotations", "xmls"),
																				databaseName = "unit_test")

	def tearDown(self):
		pass

	# def test_reduceDatasetByRois(self):
	# 	# Reduce dataset by grouping ROIs into smaller frames
	# 	self.imda.reduceDatasetByRois(offset = 1032,
	# 		outputDirectory = os.path.join(os.getcwd(), "tests", "outputs", "localization"))
	# 	# Assert the number of reduced images.
	# 	# self.assertEqual()

	def test_reduceImageDataPointByRoi(self):
		img_name = os.path.join(os.getcwd(), "tests", "localization", "images", "cv.jpg")
		xml_name = os.path.join(os.getcwd(), "tests", "localization", "annotations",\
													 "xmls", "cv.xml")
		offset = 200
		output_directory = os.path.join(os.getcwd(), "tests", "outputs")
		self.imda.reduceImageDataPointByRoi(imagePath = img_name,
																	annotationPath = xml_name,
																	offset = offset,
																	outputDirectory = output_directory)

	# def test_save_img_and_xml(self):
	# 	pass
	# 	# ImageLocalizationDataset.save_img_and_xml()

	# def test_to_xml(self):
	# 	pass
	# 	# ImageLocalizationDataset.to_xml()

if __name__ == "__main__":
	unittest.main()

