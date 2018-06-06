"""
package: Images2Dataset
class: ImageLocalizationDatasetPreprocessMethods
Author: Rodrigo Loza
Description: Preprocess methods for the ImageLocalization
class.
"""
# Libraries
from interface import Interface

class ObjectDetectionDatasetStatisticsMethods(Interface):
	
	def computeBoundingBoxStats(self, saveDataFrame = None, outputDirDataFrame = None):
		"""
		Compute basic stats for the dataset's bounding boxes.
		Args:
			saveDataFrame: A boolean that defines whether to save the dataframe or not.
			outputDirDataFrame: A string that contains the path where the dataframe will
													be saved.
		Returns:
			None
		"""
		pass



	
