"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Unit tests for ObjectDetectionDataset.
"""
import os
import unittest
from ObjectDetectionDataset import *
from Util import *

class ObjectDetectionDataset_test(unittest.TestCase):
	
	def setUp(self):
		self.imgs = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "images")
		self.annts = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "annotations", "xmls")
		self.imda = ObjectDetectionDataset(imagesDirectory = self.imgs,
																				annotationsDirectory = self.annts,
																				databaseName = "unit_test")

	def tearDown(self):
		pass

	def test_bounding_boxes(self):
		outputDirectory = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "bounding_boxes")
		os.system("rm -r {}".format(outputDirectory))
		os.mkdir("../../cars_dataset/bounding_boxes")
		self.imda.saveBoundingBoxes(outputDirectory = outputDirectory, filterClasses = ["pedestrian", "car"])

	def test_reduceDatasetByRois(self):
		outputImageDirectory = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "images_reduced")
		outputAnnotationDirectory = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "annotations_reduced", "xmls")
		os.system("rm -r {}".format(outputImageDirectory))
		os.system("rm -r {}".format(os.path.join(os.getcwd(), "../", "../", "cars_dataset", "annotations_reduced")))
		os.mkdir("../../cars_dataset/images_reduced/")
		os.mkdir("../../cars_dataset/annotations_reduced/")
		os.mkdir("../../cars_dataset/annotations_reduced/xmls/")
		# Reduce dataset by grouping ROIs into smaller frames.
		self.imda.reduceDatasetByRois(offset = [300, 300],
																	outputImageDirectory = outputImageDirectory,
																	outputAnnotationDirectory = outputAnnotationDirectory)

	def test_reduceImageDataPointByRoi(self):
		outputImageDirectory = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "images_reduced_single")
		outputAnnotationDirectory = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "annotations_reduced_single", "xmls")
		os.system("rm -r {} {}".format(outputImageDirectory, os.path.join(os.getcwd(), "../", "../", "cars_dataset", "annotations_reduced_single")))
		os.mkdir("../../cars_dataset/images_reduced_single/")
		os.mkdir("../../cars_dataset/annotations_reduced_single/")
		os.mkdir("../../cars_dataset/annotations_reduced_single/xmls/")
		offset = 300
		for i in range(1):
			for each in os.listdir(self.imgs):
				if True:
					# Get extension
					extension = Util.detect_file_extension(each)
					if (extension == None):
						raise ValueError("Extension not supported.")
					img_name = os.path.join(self.imgs, each)
					xml_name = os.path.join(self.annts, each.split(extension)[0]+".xml")
					self.imda.reduceImageDataPointByRoi(imagePath = img_name,
																				annotationPath = xml_name,
																				offset = [offset, offset],
																				outputImageDirectory = outputImageDirectory,
																				outputAnnotationDirectory = outputAnnotationDirectory)
			offset += 250

	def test_applyDataAugmentation(self):
		outputImageDirectory = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "images_augmented")
		outputAnnotationDirectory = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "annotations_augmented", "xmls")
		os.system("rm -r {}".format(outputImageDirectory))
		os.system("rm -r {}".format(os.path.join(os.getcwd(), "../", "../", "cars_dataset", "annotations_augmented")))
		os.mkdir("../../cars_dataset/images_augmented/")
		os.mkdir("../../cars_dataset/annotations_augmented/")
		os.mkdir("../../cars_dataset/annotations_augmented/xmls/")
		aug_file = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "aug_configuration_cars.json")
		self.imda.applyDataAugmentation(configurationFile = aug_file,
																outputImageDirectory = outputImageDirectory,
																outputAnnotationDirectory = outputAnnotationDirectory)


if __name__ == "__main__":
	unittest.main()

