"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Methods that apply a type of data augmentation.
"""

import numpy as np
try:
	from .BoundingBoxAugmenters import *
except:
	from BoundingBoxAugmenters import *

try:
	from .ColorAugmenters import *
except:
	from ColorAugmenters import *

try:
	from .GeometricAugmenters import *
except:
	from GeometricAugmenters import *

bndboxAugmenter = BoundingBoxAugmenters()
colorAugmenter = ColorAugmenters()
geometricAugmenter = GeometricAugmenters()

def applyGeometricAugmentation(frame = None, augmentationType = None, parameters = None):
	"""
	Applies a geometric augmentation making sure all the parameters exist or are 
	correct.
	Args:
		augmentationType: A string that contains a type of augmentation.
		parameters: A hashmap that contains parameters for the respective type 
							of augmentation.
	Returns:
		A tensor that contains a frame with the respective transformation.
	"""
	# Logic
	if (augmentationType == "scale"):
		# Apply scaling
		if (not ("size" in parameters)):
			raise Exception("ERROR: Scale requires parameter size.")
		if (not ("interpolationMethod" in parameters)):
			print("WARNING: Interpolation method for scale will be set to default value.")
		frame = geometricAugmenter.scale(frame = frame,
									size = parameters["size"],
									interpolationMethod = parameters["interpolationMethod"])
	elif (augmentationType == "crop"):
		# Apply crop
		if (not ("size" in parameters)):
			parameters["size"] = None
			print("WARNING: Size for crop will be set to default value.")
		frame = geometricAugmenter.crop(frame = frame,
																size = parameters["size"])
	elif (augmentationType == "translate"):
		# Apply pad
		if (not ("offset" in parameters)):
			raise Exception("Pad requires parameter offset.")
		bndboxes = geometricAugmenter.translate(frame = frame,
																	offset = parameters["offset"])
	elif (augmentationType == "jitterBoxes"):
		# Apply jitter boxes
		if (not ("size" in parameters)):
			raise Exception("JitterBoxes requires parameter size.")
		if (not ("quantity" in parameters)):
			parameters["quantity"] = 10
			print("WARNING: Quantity for jitter boxes will be set to its default value.")
		if (not ("color" in parameters)):
			parameters["color"] = [255,255,255]
			print("WARNING: Color for jitter boxes will be set to its default value.")
		frame = geometricAugmenter.jitterBoxes(frame = frame,
																				size = parameters["size"],
																				quantity = parameters["quantity"],
																				color = parameters["color"])
	elif (augmentationType == "horizontalFlip"):
		# Apply horizontal flip
		frame = geometricAugmenter.horizontalFlip(frame = frame)
	elif (augmentationType == "verticalFlip"):
		# Apply vertical flip
		frame = geometricAugmenter.verticalFlip(frame = frame)
	elif (augmentationType == "rotation"):
		# Apply rotation
		if (not ("theta" in parameters)):
			theta = None
			#raise Exception("ERROR: Rotation requires parameter theta.")
		else:
			theta = parameters["theta"]
		frame = geometricAugmenter.rotation(frame = frame,
																			bndbox = [0, 0, frame.shape[1], frame.shape[0]],
																			theta = theta)
	return frame

def applyColorAugmentation(frame = None, augmentationType = None, parameters = None):
	"""
	Applies a color augmentation making sure all the parameters exist or are 
	correct.
	Args:
		augmentationType: A string that contains a type of augmentation.
		parameters: A hashmap that contains parameters for the respective type 
							of augmentation.
	Returns:
		A tensor that contains a frame with the respective transformation.
	"""
	# Logic.
	if (augmentationType == "invertColor"):
		if (not ("CSpace" in parameters)):
			parameters["CSpace"] = None
		frame = colorAugmenter.invertColor(frame = frame, \
																				CSpace = parameters["CSpace"])
	elif (augmentationType == "histogramEqualization"):
		if (not ("equalizationType" in parameters)):
			parameters["equalizationType"] = None
		frame = colorAugmenter.histogramEqualization(frame = frame, \
															equalizationType = parameters["equalizationType"])
	elif (augmentationType == "changeBrightness"):
		if (not ("coefficient" in parameters)):
			raise AttributeError("coefficient for changeBrightness must be specified.")
		frame = colorAugmenter.changeBrightness(frame = frame, \
																				coefficient = parameters["coefficient"])
	elif (augmentationType == "sharpening"):
		if (not ("weight" in parameters)):
			parameters["weight"] = None
		frame = colorAugmenter.sharpening(frame = frame, weight = parameters["weight"])
	elif (augmentationType == "addGaussianNoise"):
		if (not ("coefficient" in parameters)):
			parameters["coefficient"] = None
		frame = colorAugmenter.addGaussianNoise(frame = frame, \
																				coefficient = parameters["coefficient"])
	elif (augmentationType == "gaussianBlur"):
		if (not ("sigma" in parameters)):
			parameters["sigma"] = None
		if (not ("kernelSize" in parameters)):
			parameters["kernelSize"] = None
		frame = colorAugmenter.gaussianBlur(frame = frame, \
						kernelSize = parameters["kernelSize"], sigma = parameters["sigma"])
	elif (augmentationType == "averageBlur"):
		if (not ("kernelSize" in parameters)):
			parameters["kernelSize"] = None
		frame = colorAugmenter.averageBlur(frame = frame, \
																					kernelSize = parameters["kernelSize"])
	elif (augmentationType == "medianBlur"):
		if (not ("coefficient" in parameters)):
			parameters["coefficient"] = None
		frame = colorAugmenter.medianBlur(frame = frame, \
																				coefficient = parameters["coefficient"])
	elif (augmentationType == "bilateralBlur"):
		if (not ("d" in parameters)):
			parameters["d"] = None
		if (not ("sigmaColor" in parameters)):
			parameters["sigmaColor"] = None
		if (not ("sigmaSpace" in parameters)):
			parameters["sigmaSpace"] = None
		frame = colorAugmenter.bilateralBlur(frame = frame, d = parameters["d"], \
			sigmaColor = parameters["sigmaColor"], sigmaSpace = parameters["sigmaSpace"])
	elif (augmentationType == "shiftColors"):
		frame = colorAugmenter.shiftColors(frame = frame)
	elif (augmentationType == "fancyPCA"):
		frame = colorAugmenter.fancyPCA(frame = frame)
	else:
		raise Exception("Color augmentation type not supported: {}."\
										.format(augmentationType))
	# Return result
	return frame

def applyBoundingBoxAugmentation(frame = None, boundingBoxes = None, augmentationType = None, parameters = None):
	"""
	Applies a bounding box augmentation making sure all the parameters exist or are 
	correct.
	Args:
		boundingBoxes: A list of lists of integers that contains coordinates.
		augmentationType: A string that contains a type of augmentation.
		parameters: A hashmap that contains parameters for the respective type 
							of augmentation.
	Returns:
		A tensor that contains a frame with the respective transformation.
	"""
	# Local variables.
	bndboxes = boundingBoxes
	# Logic.
	if (augmentationType == "scale"):
		# Apply scaling.
		if (not ("size" in parameters)):
			raise Exception("Scale requires parameter size.")
		if (not ("zoom" in parameters)):
			parameters["zoom"] = None
		if (not ("interpolationMethod" in parameters)):
			parameters["interpolationMethod"] = None
		frame, bndboxes = bndboxAugmenter.scale(frame = frame,
									boundingBoxes = boundingBoxes,
									size = parameters["size"],
									zoom = parameters["zoom"],
									interpolationMethod = parameters["interpolationMethod"])
	elif (augmentationType == "crop"):
		# Apply crop.
		if (not ("size" in parameters)):
			parameters["size"] = None
		bndboxes = bndboxAugmenter.crop(boundingBoxes = boundingBoxes,
									size = parameters["size"])
	elif (augmentationType == "pad"):
		# Apply pad.
		if (not ("size" in parameters)):
			raise Exception("Pad requires parameter size.")
		bndboxes = bndboxAugmenter.pad(boundingBoxes = boundingBoxes,
																	frameHeight = frame.shape[0],
																	frameWidth = frame.shape[1],
																	size = parameters["size"])
	elif (augmentationType == "jitterBoxes"):
		# Apply jitter boxes.
		if (not ("size" in parameters)):
			raise Exception("JitterBoxes requires parameter size.")
		if (not ("quantity" in parameters)):
			parameters["quantity"] = None
		frame = bndboxAugmenter.jitterBoxes(frame = frame,
																				boundingBoxes = boundingBoxes,
																				size = parameters["size"],
																				quantity = parameters["quantity"])
	elif (augmentationType == "horizontalFlip"):
		# Apply horizontal flip.
		frame = bndboxAugmenter.horizontalFlip(frame = frame,
																					boundingBoxes = boundingBoxes)
	elif (augmentationType == "verticalFlip"):
		# Apply vertical flip.
		frame = bndboxAugmenter.verticalFlip(frame = frame,
																				boundingBoxes = boundingBoxes)
	elif (augmentationType == "rotation"):
		# Apply rotation.
		if (not ("theta" in parameters)):
			theta = None
			#raise Exception("ERROR: Rotation requires parameter theta.")
		else:
			theta = parameters["theta"]
		frame = bndboxAugmenter.rotation(frame = frame,
																			boundingBoxes = boundingBoxes,
																			theta = theta)
	elif (augmentationType == "dropout"):
		# Apply dropout.
		if (not ("size" in parameters)):
			raise Exception("Dropout requires parameter size.")
		if (not ("threshold" in parameters)):
			parameters["threshold"] = None
		frame = bndboxAugmenter.dropout(frame = frame,
																	boundingBoxes = boundingBoxes,
																	size = parameters["size"],
																	threshold = parameters["threshold"])
	return frame, bndboxes

