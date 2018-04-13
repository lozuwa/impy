"""
package: impy
class: Images2Dataset (main)
Author: Rodrigo Loza
Description: Unittests for the Images2Dataset class.
"""
import unittest
from Images2Dataset import *

class Images2Dataset_test(unittest.TestCase):

	def setUp(self):
		self.im2da = Images2Dataset(dbFolder = 
																os.path.join(os.getcwd(), "db"))

	def tearDown(self):
		pass

	def test_instanceofClassIsNone(self):
		with self.assertRaises(ValueError):
			self.im2da = Images2Dataset(dbFolder =  None)

	def test_mainPathExists(self):
		with self.assertRaises(Exception):
			dummyPath = os.path.join(os.getcwd(), "f0ld3rSuperR4nD0m")
			self.im2da = Images2Dataset(dbFolder =  dummyPath)

	def test_getFolderAndFiles(self):
		with self.assertRaises(ValueError):
			self.im2da.getFolderAndFiles(parentFolder = None)

	def test_formatDataset(self):
		with self.assertRaises(Exception):
			self.im2da.getFolderAndFiles(parentFolder = None)

	def test_filterAllowedImageFormats(self):
		files = ["image1.jpg", "image2.png", "image3.tiff", "mmm.hhh"] 
		self.assertEqual(self.im2da.filterAllowedImageFormats(files),
												["image1.jpg", "image2.png"])

	def test_formatDataset(self):
		with self.assertRaises(Exception):
			im2da = Images2Dataset(dbFolder = 
																os.path.join(os.getcwd(), "db"))
			im2da["class0"].append(["phantomIm4g3.jpg"])
			im2da.formatDataset()

if __name__ == "__main__":
	unittest.main()
