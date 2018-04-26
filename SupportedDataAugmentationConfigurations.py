import os

class SupportedDataAugmentationConfigurations(object):
	"""docstring for SupportedDataAugmentationConfigurations"""
	def __init__(self):
		super(SupportedDataAugmentationConfigurations, self).__init__()
		# Hardcoded configurations
		# Types of data augmenters
		self.confAugBndbxs = "bounding_box_augmenters"
		self.confAugGeometric = "image_geometric_augmenters"
		self.confAugColor = "image_color_augmenters"
		self.confAugMultiple = "multiple_image_augmentations"
		self.supportedDataAugmentationTypes = [self.confAugMultiple, self.confAugBndbxs, self.confAugGeometric, self.confAugColor]
		# Augmenter methods for bounding boxes
		self.scale = "scale"
		self.crop = "crop"
		self.pad = "pad"
		self.jitterBoxes = "jitterBoxes"
		self.horizontalFlip = "horizontalFlip"
		self.verticalFlip = "verticalFlip"
		self.rotation = "rotation"
		self.dropout = "dropout"
		self.methodsBoundingBoxes = [self.scale, self.crop, self.pad, self.jitterBoxes, \
																self.horizontalFlip, self.verticalFlip, self.rotation, \
																self.dropout]
		self.invertColor = "invertColor"
		self.histogramEqualization = "histogramEqualization"
		self.changeBrightness = "changeBrightness"
		self.sharpening = "sharpening"
		self.addGaussianNoise = "addGaussianNoise"
		self.gaussianBlur = "gaussianBlur"
		self.shiftColors = "shiftColors"
		self.fancyPCA = "fancyPCA"
		self.methodsColors = [self.invertColor, self.histogramEqualization, self.changeBrightness, \
													self.sharpening, self.addGaussianNoise, self.gaussianBlur, \
													self.shiftColors, self.fancyPCA]

	@property
	def dataAugTypes(self):
		return self.supportedDataAugmentationTypes

	@property
	def confMultiple(self):
		return self.confAugMultiple

	@property
	def confBndbxs(self):
		return self.confAugBndbxs

	@property
	def confGeometric(self):
		return self.confAugGeometric

	@property
	def confColor(self):
		return self.confAugColor

	@property
	def methodsBndbxs(self):
		return self.methodsBoundingBoxes

	@property 
	def methodsColor(self):
		return self.methodsColors