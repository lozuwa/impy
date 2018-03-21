"""
package: Images2Dataset
class: DataAugmentation
Author: Rodrigo Loza
Description: Common data augmentation operations 
for an image.
Log:
	Novemeber, 2017 -> Re-estructured class.
	December, 2017 -> Researched most used data augmentation techniques.
	March, 2018 -> Coded methods.
TODO:
	Copy the code from the work's pc for the cropping methods.
"""
# Libraries
from interface import implements
import math
import random
import cv2

try:
	from .BoundingBoxDataAugmentationMethods import *
except:
	from BoundingBoxDataAugmentationMethods import *

class DataAugmentation(implements(BoundingBoxDataAugmentationMethods)):
	"""
	DataAugmentation class. This class implements a set of data augmentation
	tools for bouding boxes and images.
	IMPORTANT
	- This class assumes input images are numpy tensors that follow the opencv
	color format BGR.
	"""
	def __init__(self):
		# Super class
		super(DataAugmentation, self).__init__()

	def centerCrop(self,
								frameHeight = None,
								frameWidth = None,
								bndbxCoordinates = None,
								offset = None):
		"""
		Crops the bounding box from the frame giving it a symetric 
		offset at each side so it is maintained in the center.
		Args:
			frameHeight: An int that represents the height of the frame.
			frameWidth: An int that representst he width of the frame.
			bndbxCoordinates: A tuple of ints that contains the coordinates of the
												bounding box in the frame.
			offset: An int that contains the amount of space to give at each side 
							of the bounding box.
		Returns:
			An 8-sized tuple that contains the coordinates to crop the original frame
			and the new coordinates of the bounding box inside the cropped patch.
		Example:
			Given an image and a bounding box, crop the bndbox with an offset to 
			maintain context of the object.
				---------------
				|             |
				|     ---     |
				|     | |     |
				|     ---     |
				---------------
			So,
				---------------
				|   --------  |
				|   | ---   | |
				|   | | |   | |
				|   | ---   | |
				---------------
		"""
		# Local variables.
		xmin, ymin, xmax, ymax = bndbxCoordinates
		# print("c: ", xmin, ymin, xmax, ymax)
		# Initialize variables.
		heightRoi = ymax - ymin
		widthRoi = xmax - xmin
		# How much are we missing on each side?
		if heightRoi >= offset:
			missingUp = 10
			missingDown = 10
		else:
			missingY = offset - heightRoi
			missingUp = missingY // 2
			missingDown = missingY - missingUp
		if widthRoi >= offset:
			missingLeft = 10
			missingRight = 10
		else:
			missingX = offset - widthRoi
			missingLeft = missingX // 2
			missingRight = missingX - missingLeft
		# Initialize new ROI points.
		ROIxmin, ROIxmax, ROIymin, ROIymax = 0, 0, 0, 0
		# Compute the points.
		# Fox x
		if (xmin - missingLeft) <= 0:
			# Xmin is determinable.
			ROIxmin = 0
			xmin = xmin
			# xmax is just the width plus xmin.
			xmax = xmin + widthRoi
			ROIxmax = xmax + missingRight
		else:
			# xmin is determinable
			ROIxmin = xmin - missingLeft
			xmin = missingLeft
			# Then, ROIxmax might be constrained.
			if (xmax + missingRight) >= frameWidth:
				ROIxmax = frameWidth
			else:
				ROIxmax = xmax + missingRight
			# xmax is easily determinable.
			xmax = xmin + widthRoi
		# Fox y
		if (ymin - missingUp) <= 0:
			# Ymin is determinable.
			ROIymin = 0
			ymin = ymin
			# Ymax is just the heigh plus ymin.
			ymax = ymin + heightRoi
			ROIymax = ymax + missingDown
		else:
			# Ymin is determinable.
			ROIymin = ymin - missingUp
			ymin = missingUp
			# Then, ROIymax might be constrained.
			if (ymax + missingDown) >= frameHeight:
				ROIymax = frameHeight
			else:
				ROIymax = ymax + missingDown
			# Ymax is easily determinable.
			ymax = ymin + heightRoi
		assert xmin < xmax, "Xmin is bigger"
		assert ymin < ymax, "Ymin is bigger"
		# Return cropping and bndbox coordinates
		return (ROIxmin, ROIymin, ROIxmax, ROIymax,\
				xmin, ymin, xmax, ymax)

	def cropWithTranslation(self, frameHeight = None, frameWidth = None, bndbxCoordinates = None, offset = None):
		"""
		Crops the bounding box from the frame giving it an asymetric 
		offset at each side so translation is simulated inside the context
		of the cropped patch.
		Args:
			frameHeight: An int that represents the height of the frame.
			frameWidth: An int that representst he width of the frame.
			bndbxCoordinates: A tuple of ints that contains the coordinates of the
												bounding box in the frame.
			offset: An int that contains the amount of space that will be used to limit
							the size of the cropped patch.
		Returns:
			An 8-sized tuple that contains the coordinates to crop the original frame
			and the new coordinates of the bounding box inside the cropped patch.
		Example:
			Given an image and a bounding box, crop the bndbox with an offset to 
			maintain context of the object.
				-------------------
				|                 |
				|     ---         |
				|     | |         |
				|     ---         |
				|                 |
				|                 |
				|                 |           
				-------------------
			So,
				-------------------
				|   ---------     |
				|   | ---   |     |
				|   | | |   |     |
				|   | ---   |     |
				|   |       |     |
				|   ---------     |
				|                 |           
				-------------------
		"""
		# Local variables
		# Decode bndbox
		xmin, ymin, xmax, ymax = bndbxCoordinates
		# Compensation
		compensateX, compensateY = 0, 0
		# Save for later
		xmax_, xmin_, ymax_, ymin_ = xmax, xmin, ymax, ymin
		# Size of the bounding box
		widthRoi = xmax - xmin
		heightRoi = ymax - ymin
		
		# How much are we missing on each side
		# If the width is bigger than the pretended image size, then 
		# add a small amount of space.
		if widthRoi >= offset:
			missingX = 10
		# Otherwise, save the amount of space that we require.
		else:
			missingX = offset - widthRoi
		# If the height is bigger than the pretended image size, then
		# add a small amount of space.
		if heightRoi >= offset:
			missingY = 10
		else:
			missingY = offset - heightRoi

		# Define how much to add on each side of the bounding box.
		spaceXMin = int(random.random()*(missingX-5))
		spaceXMax = missingX - spaceXMin
		spaceYMin = int(random.random()*(missingY-5))
		spaceYMax = missingY - spaceYMin
		# print("missingX: {}, missingY: {}".format(missingX, missingY))
		# print("spaceXMin: {} spaceXMax: {} spaceYMin: {} spaceYMax: {}"/
		# .format(spaceXMin, spaceXMax, spaceYMin, spaceYMax))
		
		# Compute the roi and the bounding box coordinates.
		# X
		# If there is space on the left, then crop.
		if (xmin - spaceXMin) > 0:
			# Crop on the left, no restriction.
			ROIxmin = xmin - spaceXMin
			# Xmin is the space on the left.
			xmin = spaceXMin
			# That means xmax may or may not be out of bounds.
			# If ROIxmax is out of bounds, then crop at width.
			if (xmax + spaceXMax) >= frameWidth:
				# Crop at width.
				ROIxmax = frameWidth
				# Reduce ROIxmin and xmin to compensate.
				widthRoiC = ROIxmax - ROIxmin
				if (offset - widthRoiC) >= 0:
					compensateX = (offset - widthRoiC)
				else:
					compensateX = 0
				ROIxmin -= compensateX
				xmin += compensateX
			# If xmax is not out of bounds, then add xmax and spaceXMax.
			else:
				ROIxmax = xmax + spaceXMax
			# Xmax is determinable
			xmax = xmin + widthRoi
		# If there is not space on the left, then crop at origin.
		else:
			# Crop at origin.
			ROIxmin = 0
			# Xmin is maintained
			xmin = xmin
			# ROIxmax and xmax are determinable
			ROIxmax = xmax + spaceXMax 
			# Compensate in xmax
			widthRoiC = ROIxmax - ROIxmin
			if (offset - widthRoiC) >= 0:
				compensateX = (offset - widthRoiC)
			else:
				compensateX = 0
			ROIxmax += compensateX
			xmax = xmin + widthRoi

		# Y
		# If there is space on the top, then crop.
		if (ymin - spaceYMin) > 0:
			# Crop on the top, no restriction.
			ROIymin = ymin - spaceYMin
			# Ymin should be the spaceYMin.
			ymin = spaceYMin
			# That means ROIymax may or may not be out of bounds.
			# If ROIymax is out of bounds.
			if (ymax + spaceYMax) >= frameHeight:
				# Crop at height.
				ROIymax = frameHeight
				heightRoiC = ROIymax - ROIymin
				# Compensate ROIymin.
				if (offset - heightRoiC) >= 0:
					compensateY = offset - heightRoiC
				else:
					compensateY = 0
				ROIymin -= compensateY
				ymin += compensateY
			# Otherwise, ROIymax is ymax + spaceYMax
			else:
				ROIymax = ymax + spaceYMax
			# Ymax should be determinable.
			ymax = ymin + heightRoi
		# If there is no space on the top, then crop at origin.
		else:
			# Crop at origin.
			ROIymin = 0
			# Ymin is maintained.
			ymin = ymin
			# ROIYmax and ymax are determinable.
			ROIymax = ymax + spaceYMax
			# Compensate in ROIymax.
			heightRoiC = ROIymax - ROIymin
			if (offset - heightRoiC):
				compensateY = offset - heightRoiC
			else:
				compensateY = 0
			ROIymax += compensateY
			# ymax is determinable.
			ymax = ymin + heightRoi

		# print("Before: {},{},{},{}".format(xmin_, xmax_,\
		# 																		ymin_, ymax_))
		# print("ROI: {}, {}, {}, {}".format(ROIxmax, ROIxmin,\
		# 																	ROIymax, ROIymin))
		# print("Compensate: x: {}, y: {}".format(compensateX, compensateY))
		# print("Coordinates: {},{},{},{}".format(xmin, xmax,\
		# 																				ymin, ymax))

		assert (ROIxmax - ROIxmin) > offset - (offset*0.2), "X dimension is too small"
		assert (ROIymax - ROIymin) > offset - (offset*0.2), "Y dimension is too small"

		assert (ROIxmax - ROIxmin) < offset + (offset*0.2), "X dimension is too big"
		assert (ROIymax - ROIymin) < offset + (offset*0.2), "Y dimension is too big"

		square_relation = ((ROIxmax - ROIxmin) / (ROIymax - ROIymin))
		assert (square_relation > 0.90) and (square_relation < 1.10), "ROI is not a square"

		assert ymin >= 0, "Ymin is negative"
		assert ymax >= 0, "Ymax is negative"
		assert xmin >= 0, "Xmin is negative"
		assert xmax >= 0, "Xmax is negative"

		assert (ymax-ymin) == (ymax_-ymin_), "Heights are not the same : {}, {} {}".\
											format((ymax-ymin), ymax_, ymin_)
		assert (xmax-xmin) == (xmax_-xmin_), "Widths are not the same : {}, {} {}".\
											format((xmax-xmin), xmax_, xmin_)

		# Return the computed coordinates
		return (ROIxmin, ROIxmax, ROIymin, ROIymax,\
						xmin, xmax, ymin, ymax)

	def jitterBoxes(self, frame = None, quantity = None):
		"""
		Creates random jitter boxes inside a bounding box cropped from its image.
		Args:
			frame: A tensor that contains an image.
			quantity: An int that tells how many jitter boxes to create inside 
							the frame.
		Returns:
			A tensor that contains an image altered by jitter boxes.
		"""
		# Assertions
		try:
			if frame == None:
				raise Exception("Frame cannot be empty.")
		except:
			pass
		if quantity == None:
			quantity = 10
		# Local variables
		frameSize = frame.shape
		rows, cols = frameSize[0], frameSize[1]
		xj, yj = frameSize[0] // 8, frameSize[0] // 8
		# Create boxes
		for i in range(quantity):
			y = int(random.random() * rows) - rows // 3
			x = int(random.random() * cols) - cols // 3
			# Draw boxes on top of the image
			frame = cv2.rectangle(frame, (x, y), (x+xj, y+yj), (0, 0, 0), -1)
		# Return frame
		return frame

	def horizontalFlip(self, frame = None):
		"""
		Flip a bouding box by its horizontal axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
		Returns:
			A tensor that has been flipped by its horizontal axis.
		"""
		# Flip
		frame = cv2.flip(frame, 0)
		return frame

	def verticalFlip(self, frame = None):
		"""
		Flip a bouding box by its vertical axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
		Returns:
			A tensor that has been flipped by its vertical axis.
		"""
		# Flip
		frame = cv2.flip(frame, 1)
		return frame

	def randomRotation(self, frame = None, bndbox = None, theta = None):
		"""
		Rotate a frame clockwise by random degrees. Random degrees
		is a number that is between 20-360.
		Args:
			frame: A tensor that contains an image.
			bndbox: A tuple that contains the ix, iy, x, y coordinates 
							of the bounding box in the image.
			theta: An int that contains the amount of degrees to move.
							Default is random.
		Returns:
			A tensor that contains the rotated image and a tuple
			that contains the rotated coordinates of the bounding box.
		"""
		# Assertions
		try:
			if frame == None:
				raise Exception("Frame cannot be emtpy.")
		except:
			pass
		if bndbox == None:
			raise Exception("Bnbdbox cannot be empty")
		if theta == None:
			theta = (random.random() * math.pi) + math.pi / 3
		# Local variables
		thetaDegrees = theta * 180 / math.pi
		rows, cols, depth = frame.shape
		# print("Degrees: ", thetaDegrees)
		# print("Rows, cols: ", rows//2, cols//2)
		# Decode the bouding box
		ix, iy, x, y = bndbox
		# print("Original: ", bndbox)
		# Fix the y coordinate since matrix transformations
		# assume 0,0 is at the left bottom corner.
		iy, y = rows-iy, rows-y
		# print(ix, iy, x, y)
		# Center the coordinates with respect to the 
		# center of the image.
		ix, iy, x, y = ix-(cols//2), iy-(rows//2), x-(cols//2), y-(rows//2)
		# print("Centered: ", ix, iy, x, y)
		# print(ix, iy, x, y)
		# Write down coordinates
		p0 = [ix, iy]
		p1 = [x, iy]
		p2 = [ix, y]
		p3 = [x, y]
		# Compute rotations on coordinates
		p0[0], p0[1] = DataAugmentation.rotation_equations(p0[0], p0[1], theta)
		p1[0], p1[1] = DataAugmentation.rotation_equations(p1[0], p1[1], theta)
		p2[0], p2[1] = DataAugmentation.rotation_equations(p2[0], p2[1], theta)
		p3[0], p3[1] = DataAugmentation.rotation_equations(p3[0], p3[1], theta)
		# Add centers to compensate
		p0[0], p0[1] = p0[0] + (cols//2), rows - (p0[1] + (rows//2))
		p1[0], p1[1] = p1[0] + (cols//2), rows - (p1[1] + (rows//2))
		p2[0], p2[1] = p2[0] + (cols//2), rows - (p2[1] + (rows//2))
		p3[0], p3[1] = p3[0] + (cols//2), rows - (p3[1] + (rows//2))
		# Rotate image
		M = cv2.getRotationMatrix2D((cols/2, rows/2), thetaDegrees, 1)
		frame = cv2.warpAffine(frame, M, (cols, rows))
		xs = [p0[0], p1[0], p2[0], p3[0]]
		ys = [p0[1], p1[1], p2[1], p3[1]]
		ix, x = min(xs), max(xs)
		iy, y = min(ys), max(ys)
		# print(p0, p1, p2, p3)
		# Return frame and coordinates
		return frame, [ix, iy, x, y]

	@staticmethod
	def rotation_equations(x, y, theta):
		"""
		Apply a 2D rotation matrix to a 2D coordinate by theta degrees.
		Args:
			x: An int that represents the x dimension of the coordinate.
			y: An int that represents the y dimension of the coordinate.
		"""
		x_result = int((x*math.cos(theta)) - (y*math.sin(theta)))
		y_result = int((x*math.sin(theta)) + (y*math.cos(theta)))
		return x_result, y_result

	def addRandomBlur(self, frame = None, sigma = None):
		"""
		Blur an image applying a gaussian filter with a random sigma(0, sigma_max)
		Sigma's default value is between 1 and 3.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor with a rotation of the original image.
		"""
		# Assertions
		try:
			if frame == None:
				raise Exception("Frame cannot be emtpy.")
		except:
			pass
		if sigma == None:
			sigma = float(random.random()*3) + 1
		# Apply gaussian filter
		frame = cv2.GaussianBlur(frame, (5, 5), sigma)
		return frame

	def shiftColors(self, frame = None):
		"""
		Shifts the colors of the frame.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has shifted the order of its colors.
		"""
		# Assertions
		try:
			if frame == None:
				raise Exception("Frame cannot be emtpy.")
		except:
			pass
		if frame.shape != 3:
			raise Exception("Frame must have 3 dimensions")
		# Local variables
		colorsOriginal = [0, 1, 2]
		colorsShuffle = colorsOriginal
		# Shuffle list of colors
		while(colorsOriginal == colorsShuffle):
			shuffle(colorsShuffle)
		# Swap color dimensions
		frame[:, :, 0], frame[:, :, 1], \
		frame[:, :, 2] = frame[:, :, colorsShuffle[0]], \
										frame[:, :, colorsShuffle[1]], \
										frame[:, :, colorsShuffle[2]]
		return frame


