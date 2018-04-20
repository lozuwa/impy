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
		self.imda = ImageLocalizationDataset(images = os.path.join(os.getcwd(), \
																									"tests", "localization", \
																									"images"),
																				annotations = os.path.join(os.getcwd(), \
																										"tests", "localization", \
																										"annotations", "xmls"),
																				databaseName = "unit_test")

	def tearDown(self):
		pass

	def test_reduceDatasetByRois(self):
		os.system("rm -r {}".format(os.path.join(os.getcwd(), "tests", "outputs", "localization")))
		# Reduce dataset by grouping ROIs into smaller frames
		self.imda.reduceDatasetByRois(offset = 1032,
			outputDirectory = os.path.join(os.getcwd(), "tests", "outputs", "localization"))
		# Assert the number of reduced images.
		# self.assertEqual()

	def test_reduceImageDataPointByRoi(self):
		output_directory = os.path.join(os.getcwd(), "tests", "outputs")
		os.system("rm {}/images/* {}/annotations/xmls/*".format(output_directory, output_directory))
		offset = 300
		for i in range(1):
			for each in os.listdir(os.path.join(os.getcwd(), "tests", "localization", "images")):
				if True:#each.endswith(".png"):
					# Get extension
					extension = Util.detect_file_extension(each)
					if (extension == None):
						raise ValueError("Extension not supported.")
					img_name = os.path.join(os.getcwd(), "tests", "localization", "images", each)
					xml_name = os.path.join(os.getcwd(), "tests", "localization", "annotations",\
															 "xmls", each.split(extension)[0]+".xml")
					self.imda.reduceImageDataPointByRoi(imagePath = img_name,
																				annotationPath = xml_name,
																				offset = offset,
																				outputDirectory = output_directory)
				offset += 0

	def test_applyDataAugmentation(self):
		outputDirectory = os.path.join(os.getcwd(), "tests", "outputs", "DatasetAugmentation")
		os.system("rm -r {}".format(outputDirectory))
		augmentations = os.path.join(os.getcwd(), "tests","augmentation.json")
		self.imda.applyDataAugmentation(augmentations = augmentations,
					outputDirectory = outputDirectory)

	# def test_save_img_and_xml(self):
	# 	pass
	# 	# ImageLocalizationDataset.save_img_and_xml()

	# def test_to_xml(self):
	# 	pass
	# 	# ImageLocalizationDataset.to_xml()

if __name__ == "__main__":
	unittest.main()

