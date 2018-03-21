"""
package: Images2Dataset
class: ImagePreprocessing
Author: Rodrigo Loza
Description: Common pre-processing operations 
for an image.
Log:
	August, 2017 -> Created class and added most relevant functions.
	September, 2017 -> Created more features for the DivideIntoPatches method.
	November, 2017 -> Refactored all the code to Google style and more OOP paradigm.
	December, 2017 -> Tested refactor to avoid breaking external code.
	March, 2018 -> Refactored to another structure.
"""
# Utils
import numpy as np
import cv2
import math

class ImagePreprocessing(object):
	"""
	Preprocessing operations performed on an image.
	"""
	def __init__(self):
		"""
		Constructor.
		"""
		super(ImagePreprocessing, self).__init__()
	
	def divideIntoPatches(self,
												imageWidth = None,
												imageHeight = None,
												slideWindowSize = None,
												strideSize = None,
												padding = None,
												numberPatches = None):
		"""
		Divides the image into NxM patches depending on the stride size,
		the sliding window size and the type of padding.
		Args:
			imageWidth: An int that represents the width of the image.
			imageHeight: An int that represents the height of the image.
			slideWindowSize: A tuple (width, height) that represents the size
													of the sliding window.
			strideSize: A tuple (width, height) that represents the amount
									of pixels to move on height and width direction.
			padding: A string ("VALID", "SAME", "VALID_FIT_ALL") that tells the type of
								padding.
			numberPatches: A tuple (numberWidth, numberHeight) that 
												contains the number of patches in each axis.
		Return: 
			A tuple containing the number of patches that fill the
			given parameters, an int containing the number of row patches,
			an int containing the number of column patches
		"""
		# Assertions
		if imageWidth == None:
			raise Exception("Image width cannot be empty.")
		if imageHeight == None:
			raise Exception("Image height cannot be empty.")
		if slideWindowSize == None:
			slideWindowSize = (0, 0)
		if strideSize == None:
			strideSize = (0, 0)
		if padding == None:
			padding = "VALID"
		if numberPatches == None:
			numberPatches = (1, 1)
		# Get sliding window sizes
		slideWindowWidth, slideWindowHeight = slideWindowSize[0], slideWindowSize[1]
		if (slideWindowHeight > imageHeight) or (slideWindowWidth > imageWidth):
			raise Exception("Slide window size is too big.")
		# Get strides sizes
		strideWidth, strideHeight = strideSize[0], strideSize[1]
		if (strideHeight > imageHeight) or (strideWidth > imageWidth):
			raise Exception("Stride size is too big.")
		# Start padding operation
		if padding == "VALID":
			startPixelsHeight = 0
			endPixelsHeight = slideWindowHeight
			startPixelsWidth = 0
			endPixelsWidth = slideWindowWidth
			patchesCoordinates = []
			numberPatchesHeight, numberPatchesWidth = ImagePreprocessing.get_valid_padding(slideWindowHeight,
																								 strideHeight,
																								 imageHeight,
																								 slideWindowWidth,
																								 strideWidth,
																								 imageWidth)
			print("numberPatchesHeight: ", numberPatchesHeight, "numberPatchesWidth: ", numberPatchesWidth)
			for i in range(numberPatchesHeight):
				for j in range(numberPatchesWidth):
					patchesCoordinates.append([startPixelsHeight,\
													startPixelsWidth,\
													endPixelsHeight,\
													endPixelsWidth])
					# Update width with strides
					startPixelsWidth += strideWidth
					endPixelsWidth += strideWidth
				# Re-initialize the width parameters 
				startPixelsWidth = 0
				endPixelsWidth = slideWindowWidth
				# Update height with height stride size
				startPixelsHeight += strideHeight
				endPixelsHeight += strideHeight
			return patchesCoordinates,\
					numberPatchesHeight,\
					numberPatchesWidth
		elif padding == "SAME":
			startPixelsHeight = 0
			endPixelsHeight = slideWindowHeight
			startPixelsWidth = 0
			endPixelsWidth = slideWindowWidth
			patchesCoordinates = []
			# Modify image tensor
			zeros_h, zeros_w = ImagePreprocessing.get_same_padding(slideWindowHeight,
																				 strideHeight,
																				 imageHeight,
																				 slideWindowWidth,
																				 strideWidth,
																				 imageWidth)
			imageWidth += zeros_w
			imageHeight += zeros_h
			# Valid padding stride should fit exactly
			numberPatchesHeight, numberPatchesWidth = ImagePreprocessing.get_valid_padding(slideWindowHeight,
																		 strideHeight,
																		 imageHeight,
																		 slideWindowWidth,
																		 strideWidth,
																		 imageWidth)
			for i in range(numberPatchesHeight):
				for j in range(numberPatchesWidth):
					patchesCoordinates.append([startPixelsHeight,\
													startPixelsWidth,\
													endPixelsHeight,\
													endPixelsWidth])
					# Update width with strides
					startPixelsWidth += strideWidth
					endPixelsWidth += strideWidth
				# Re-initialize the width parameters 
				startPixelsWidth = 0
				endPixelsWidth = slideWindowWidth
				# Update height with height stride size
				startPixelsHeight += strideHeight
				endPixelsHeight += strideHeight
			return patchesCoordinates,\
					numberPatchesHeight,\
					numberPatchesWidth,\
					zeros_h,\
					zeros_w

		elif padding == "VALID_FIT_ALL":
			# Get number of patches
			patchesCols = numberPatches[0]
			patchesRows = numberPatches[1]
			# Determine the size of the windows for the patches
			strideHeight = math.floor(imageHeight / patchesRows)
			slideWindowHeight = strideHeight
			strideWidth = math.floor(imageWidth / patchesCols)
			slideWindowWidth = strideWidth
			#print("Size: ", strideHeigth, slideWindowHeight, strideWidth, slideWindowWidth)
			# Get valid padding
			startPixelsHeight = 0
			endPixelsHeight = slideWindowHeight
			startPixelsWidth = 0
			endPixelsWidth = slideWindowWidth
			patchesCoordinates = []
			numberPatchesHeight, numberPatchesWidth = ImagePreprocessing.get_valid_padding(slideWindowHeight,
																		 strideHeight,
																		 imageHeight,
																		 slideWindowWidth,
																		 strideWidth,
																		 imageWidth)
			#print("numberPatchesHeight: ", numberPatchesHeight, "numberPatchesWidth: ", numberPatchesWidth)
			for i in range(numberPatchesHeight):
				for j in range(numberPatchesWidth):
					patchesCoordinates.append([startPixelsHeight,\
													startPixelsWidth,\
													endPixelsHeight,\
													endPixelsWidth])
					# Update width with strides
					startPixelsWidth += strideWidth
					endPixelsWidth += strideWidth
				# Re-initialize the width parameters
				startPixelsWidth = 0
				endPixelsWidth = strideWidth
				# Update height with height stride size
				startPixelsHeight += strideHeight
				endPixelsHeight += strideHeight
			return patchesCoordinates,\
					numberPatchesHeight,\
					numberPatchesWidth
		else:
			raise Exception("Type of padding not understood")

	@staticmethod
	def get_valid_padding(slide_window_height,
											 stride_height,
											 image_height,
											 slide_window_width,
											 stride_width,
											 image_width):
		"""
		Given the dimensions of an image, the strides of the sliding window
		and the size of the sliding window. Find the number of patches that
		fit in the image if the type of padding is VALID.
		Args:
			slide_window_height: int that represents the height of the slide
									Window.
			stride_height: int that represents the height of the stride.
			image_height: int that represents the height of the image.
			slide_window_width: int that represents the width of the slide
									window.
			stride_width: int that represents the width of the stride.
			image_width: int that represents the width of the image.
		Returns:
			A tuple containing the number of patches in the height and 
				and the width dimension.
		"""
		number_patches_height = 0
		number_patches_width = 0
		while(True):
			if slide_window_height <= image_height:
				slide_window_height += stride_height
				number_patches_height += 1
			elif slide_window_height > image_height:
				break
			else:
				continue
		while(True):
			if slide_window_width <= image_width:
				slide_window_width += stride_width
				number_patches_width += 1	
			elif slide_window_width > image_width:
				break
			else:
				continue
		return (number_patches_height, number_patches_width)

	@staticmethod
	def get_same_padding(slide_window_height,
						 stride_height,
						 image_height,
						 slide_window_width,
						 stride_width,
						 image_width):
		""" 
		Given the dimensions of an image, the strides of the sliding window
		and the size of the sliding window. Find the number of zeros needed
		for the image so the sliding window fits as type of padding SAME. 
		Then find the number of patches that fit in the image. 
		:param slideWindowHeight: int that represents the height of the slide 
									Window
		:param strideHeight: int that represents the height of the stride
		:param imageHeight: int that represents the height of the image
		:param slideWindowWidth: int that represents the width of the slide
									window
		:param strideWidth: int that represents the width of the stride
		:param imageWidth: int that represents the width of the image
		: return: a tuple containing the amount of zeros
					to add in the height dimension and the amount of zeros
					to add in the width dimension. 
		"""
		# Initialize auxiliar variables
		number_patches_height = 0
		number_patches_width = 0
		# Calculate the number of patches that fit
		while(True):
			if slide_window_height <= image_height:
				slide_window_height += stride_height
				number_patches_height += 1
			elif slide_window_height > image_height:
				break
			else:
				continue
		while(True):
			if slide_window_width <= image_width:
				slide_window_width += stride_width
				number_patches_width += 1	
			elif slide_window_width > image_width:
				break
			else:
				continue
		# Fix the excess in slide_window
		slide_window_height -= stride_height
		slide_window_width -= stride_width
		#print(number_patches_height, number_patches_width)
		#print(slide_window_height, slide_window_width)
		# Calculate how many pixels to add
		zeros_h = 0
		zeros_w = 0
		if slide_window_width == image_width:
			pass
		else:
			# Pixels left that do not fit in the kernel
			assert slide_window_width < image_width, "Slide window + stride is bigger than width"
			zeros_w = (slide_window_width + stride_width) - image_width
		if slide_window_height == image_height:
			pass
		else:
			# Pixels left that do not fit in the kernel 
			assert slide_window_height < image_height, "Slide window + stride is bigger than height"
			zeros_h = (slide_window_height + stride_height) - image_height
		#print(slide_window_height, imageHeight, resid_h, zeros_h)
		# Return amount of zeros
		return (zeros_h, zeros_w)

	@staticmethod
	def lazySAMEpad(frame,
									zeros_h,
									zeros_w,
									padding_type = "ONE_SIDE"):
		"""
		Given an image and the number of zeros to be added in height 
		and width dimensions, this function fills the image with the 
		required zeros.
		:param frame: opencv image of 3 dimensions
		:param zeros_h: int that represents the amount of zeros to be added
						in the height dimension
		:param zeros_w: int that represents the amount of zeros to be added 
						in the width dimension
		:param padding_type: string that determines the side where to pad the image.
						If BOTH_SIDES, then padding is applied to both sides.
						If ONE_SIDE, then padding is applied to the right and the bottom.
						Default: ONE_SIDE
		: return: a new opencv image with the added zeros
		"""
		if padding_type == "BOTH_SIDES":
			rows, cols, d = frame.shape
			# If height is even or odd
			if (zeros_h % 2 == 0):
				zeros_h = int(zeros_h/2)
				frame = r_[np.zeros((zeros_h, cols, 3)), frame,\
							np.zeros((zeros_h, cols, 3))]
			else:
				zeros_h += 1
				zeros_h = int(zeros_h/2)
				frame = r_[np.zeros((zeros_h, cols, 3)), frame,\
							np.zeros((zeros_h, cols, 3))]

			rows, cols, d = frame.shape
			# If width is even or odd 
			if (zeros_w % 2 == 0):
				zeros_w = int(zeros_w/2)
				# Container 
				container = np.zeros((rows,(zeros_w*2+cols),3), np.uint8)
				container[:,zeros_w:container.shape[1]-zeros_w:,:] = frame
				frame = container #c_[np.zeros((rows, zeros_w)), frame, np.zeros((rows, zeros_w))]
			else:
				zeros_w += 1
				zeros_w = int(zeros_w/2)
				container = np.zeros((rows, (zeros_w*2+cols), 3), np.uint8)
				container[:, zeros_w:container.shape[1]-zeros_w:, :] = frame
				frame = container #c_[np.zeros((rows, zeros_w, 3)), frame, np.zeros((rows, zeros_w, 3))]
			return frame
		elif padding_type == "ONE_SIDE":
			rows, cols, d = frame.shape
			# Pad height dimension
			frame = r_[frame, np.zeros((zeros_h, cols, 3))]
			# Pad width dimension
			rows, cols, d = frame.shape
			container = np.zeros((rows, cols + zeros_w, 3), np.uint8)
			container[:, :cols, :] = frame
			container[:, cols:, :] = np.zeros((rows, zeros_w, 3), np.uint8)
			return container

def drawGrid(frame,
			patches,
			patchesLabels):
	"""
	Draws the given patches on top of the input image
	:param frame: opencv input image
	:param patches: a list containing the coordinates of the patches
					calculated for the image
	: return: opencv image named frame that contains the same input
				image but with a grid of patches draw on top.
	"""
	# Iterate through patches
	for i in range(len(patches)):
		# Get patch
		patch = patches[i]
		# "Decode" patch
		startHeight, startWidth, endHeight, endWidth = patch[0], patch[1],\
														patch[2], patch[3]
		# Draw grids
		cv2.rectangle(frame, (startWidth, startHeight),\
						(endWidth, endHeight), (0, 0, 255), 12)
		roi = np.zeros([patch[2]-patch[0], patch[3]-patch[1], 3],\
						np.uint8)
		# Paint the patch
		if patchesLabels[i] == 1:
			roi[:,:,:] = (0,0,255)
		else:
			roi[:,:,:] = (0,255,0)
		cv2.addWeighted(frame[patch[0]:patch[2],patch[1]:patch[3],:],\
						0.8, roi, 0.2, 0, roi)
		frame[patch[0]:patch[2],patch[1]:patch[3],:] = roi
	return frame

def drawBoxes(frame,
			  patchesCoordinates,
			  patchesLabels):
	"""
	Draws a box or boxes over the frame
	:param frame: input cv2 image
	:param patchesCoordinates: a list containing sublists [iy, ix, y, x]
							  of coordinates
	:param patchesLabels: a list containing the labels of the coordinates
	"""
	for coord in patchesCoordinates:
		# Decode coordinate [iy, ix, y, x]
		iy, ix, y, x = coord[0], coord[1], coord[2], coord[3]
		# Draw box
		cv2.rectangle(frame, (ix, iy), (x, y), (255, 0, 0), 8)
	return frame
