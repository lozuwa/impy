"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Testing units for ImageLocalizationDataset.
"""
import os
import unittest
from ImageLocalizationDataset import *
from Util import *

class ImageLocalizationDataset_test(unittest.TestCase):
	
	def setUp(self):
		imgs = os.path.join(os.getcwd(), "tests", "localization", "images")
		annts = os.path.join(os.getcwd(), "tests", "localization", "annotations")
		self.imda = ImageLocalizationDataset(images = imgs,
																				annotations = annts,
																				databaseName = "unit_test")

	def tearDown(self):
		pass

	# def test_reduceDatasetByRois(self):
	# 	imgs = os.path.join(os.getcwd(), "tests", "outputs", "reduceByRois", "images")
	# 	annts = os.path.join(os.getcwd(), "tests", "outputs", "reduceByRois", "annotations")
	# 	os.system("rm {}/*".format(imgs))
	# 	os.system("rm {}/*".format(annts))
	# 	# Reduce dataset by grouping ROIs into smaller frames.
	# 	self.imda.reduceDatasetByRois(offset = 1032,
	# 																outputImageDirectory = imgs,
	# 																outputAnnotationDirectory = annts)

	# def test_reduceImageDataPointByRoi(self):
	# 	outputImageDirectory = os.path.join(os.getcwd(), "tests", "outputs", "images")
	# 	outputAnnotationDirectory = os.path.join(os.getcwd(), "tests", "outputs", "annotations")
	# 	os.system("rm {}/* {}/*".format(outputImageDirectory, outputAnnotationDirectory))
	# 	offset = 300
	# 	for i in range(1):
	# 		for each in os.listdir(os.path.join(os.getcwd(), "tests", "localization", "images")):
	# 			if True:#each.endswith(".png"):
	# 				# Get extension
	# 				extension = Util.detect_file_extension(each)
	# 				if (extension == None):
	# 					raise ValueError("Extension not supported.")
	# 				img_name = os.path.join(os.getcwd(), "tests", "localization", "images", each)
	# 				xml_name = os.path.join(os.getcwd(), "tests", "localization", "annotations",\
	# 															 each.split(extension)[0]+".xml")
	# 				self.imda.reduceImageDataPointByRoi(imagePath = img_name,
	# 																			annotationPath = xml_name,
	# 																			offset = offset,
	# 																			outputImageDirectory = outputImageDirectory,
	# 																			outputAnnotationDirectory = outputAnnotationDirectory)
	# 			offset += 0

	def test_applyDataAugmentation(self):
		imgs = os.path.join(os.getcwd(), "tests", "outputs", "DatasetAugmentation", "images")
		annts = os.path.join(os.getcwd(), "tests", "outputs", "DatasetAugmentation", "annotations")
		os.system("rm {}/*".format(imgs))
		os.system("rm {}/*".format(annts))
		aug_confs = [os.path.join("tests", i) for i in os.listdir(os.path.join(os.getcwd(), "tests")) if i.endswith(".json") and not "geometric" in i]
		for each in aug_confs:
			self.imda.applyDataAugmentation(configurationFile = each,
																	outputImageDirectory = imgs,
																	outputAnnotationDirectory = annts)

	# def test_save_img_and_xml(self):
	# 	pass
	# 	# ImageLocalizationDataset.save_img_and_xml()

	# def test_to_xml(self):
	# 	pass
	# 	# ImageLocalizationDataset.to_xml()

if __name__ == "__main__":
	unittest.main()

