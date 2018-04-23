import os

class SupportedDataAugmentationConfigurations(object):
	"""docstring for SupportedDataAugmentationConfigurations"""
	def __init__(self):
		super(SupportedDataAugmentationConfigurations, self).__init__()
		# Hardcoded configurations
		self.confAugBndbxs = "bounding_box_augmenters"
		self.confAugGeometric = "image_geometric_augmenters"
		self.confAugColor = "image_color_augmenters"
		self.supportedDataAugmentationTypes = ["bounding_box_augmenters", "image_augmenters"]

	@property
	def dataAugTypes(self):
		return self.supportedDataAugmentationTypes

	@property
	def confBndbxs(self):
		return self.confAugBndbxs

	@property
	def confGeometric(self):
		return self.confAugGeometric

	@property
	def confColor(self):
		return self.confAugColor
