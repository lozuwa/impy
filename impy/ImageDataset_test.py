import unittest

try:
	from .ImageDataset import *
except:
	from ImageDataset import *

class ImageDataset_test(unittest.TestCase):
	def setUp(self):
		self.imgs = os.path.join(os.getcwd(), "../", "../", "cars_dataset", "images")
		self.imda = ImageDataset(imagesDirectory = self.imgs, \
														dbName = "UnitTest")

	def tearDown(self):
		pass

	def test_apply_data_augmentation(self):
		os.system("rm {}/*".format(os.path.join(os.getcwd(), "../", "../", "cars_dataset", "images_single")))
		conf_file = os.path.join(os.getcwd(), "../", "confs_examples", \
								"aug_multiple_geometric_color_sequential.json")
		self.imda.applyDataAugmentation(configurationFile = conf_file, \
																outputImageDirectory = os.path.join(os.getcwd(), \
														"../", "../", "cars_dataset", "images_single"))

if __name__ == "__main__":
	unittest.main()