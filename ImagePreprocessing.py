"""
package: Images2Dataset
class: ImagePreprocessing
Author: Rodrigo Loza
Description: Common pre-processing operations 
for an image.
"""
# Utils
from .utils import *

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
												image_width,
												image_height,
												slide_window_size = (0, 0),
												stride_size = (0, 0),
												padding = "VALID",
												numberPatches = (1, 1)):
		"""
		Divides the image into NxM patches depending on the stride size,
		the sliding window size and the type of padding.
		Args:
			image_width: An int that represents the width of the image.
			image_height: An int that represents the height of the image.
			slide_window_size: A tuple (width, height) that represents the size
													of the sliding window.
			stride_size: A tuple (width, height) that represents the amount
									of pixels to move on height and width direction.
			padding: A string ("VALID", "SAME", "VALID_FIT_ALL") that tells the type of
								padding.
			number_of_patches: A tuple (number_width, number_height) that 
												contains the number of patches in each axis.
		Return: 
			A tuple containing the number of patches that fill the
			given parameters, an int containing the number of row patches,
			an int containing the number of column patches
		"""
		# Get sliding window sizes
		slide_window_width, slide_window_height = slide_window_size[0], slide_window_size[1]
		assert (slide_window_height < image_height) and (slide_window_width < image_width),\
				SLIDE_WINDOW_SIZE_TOO_BIG
		# Get strides sizes
		stride_width, stride_height = stride_size[0], stride_size[1]
		assert (stride_height < image_height) and (stride_width < image_width),\
				STRIDE_SIZE_TOO_BIG
		# Start padding operation
		if padding == 'VALID':
			start_pixels_height = 0
			end_pixels_height = slide_window_height
			start_pixels_width = 0
			end_pixels_width = slide_window_width
			patches_coordinates = []
			numberPatches_height, numberPatches_width = get_valid_padding(slide_window_height,
																		 stride_height,
																		 image_height,
																		 slide_window_width,
																		 stride_width,
																		 image_width)
			print('numberPatchesHeight: ', numberPatches_height, 'numberPatchesWidth: ', numberPatches_width)
			for i in range(numberPatches_height):
				for j in range(numberPatches_width):
					patches_coordinates.append([start_pixels_height,\
													start_pixels_width,\
													end_pixels_height,\
													end_pixels_width])
					# Update width with strides
					start_pixels_width += stride_width
					end_pixels_width += stride_width
				# Re-initialize the width parameters 
				start_pixels_width = 0
				end_pixels_width = slide_window_width
				# Update height with height stride size
				start_pixels_height += stride_height
				end_pixels_height += stride_height
			return patches_coordinates,\
					numberPatches_height,\
					numberPatches_width
		elif padding == 'SAME':
			start_pixels_height = 0
			end_pixels_height = slide_window_height
			start_pixels_width = 0
			end_pixels_width = slide_window_width
			patches_coordinates = []
			# Modify image tensor
			zeros_h, zeros_w = get_same_padding(slide_window_height,
																				 stride_height,
																				 image_height,
																				 slide_window_width,
																				 stride_width,
																				 image_width)
			image_width += zeros_w
			image_height += zeros_h
			# Valid padding stride should fit exactly
			numberPatches_height, numberPatches_width = get_valid_padding(slide_window_height,
																		 stride_height,
																		 image_height,
																		 slide_window_width,
																		 stride_width,
																		 image_width)
			for i in range(numberPatches_height):
				for j in range(numberPatches_width):
					patches_coordinates.append([start_pixels_height,\
													start_pixels_width,\
													end_pixels_height,\
													end_pixels_width])
					# Update width with strides
					start_pixels_width += stride_width
					end_pixels_width += stride_width
				# Re-initialize the width parameters 
				start_pixels_width = 0
				end_pixels_width = slide_window_width
				# Update height with height stride size
				start_pixels_height += stride_height
				end_pixels_height += stride_height
			return patches_coordinates,\
					numberPatches_height,\
					numberPatches_width,\
					zeros_h,\
					zeros_w

		elif padding == "VALID_FIT_ALL":
			# Get number of patches
			patches_cols = numberPatches[0]
			patches_rows = numberPatches[1]
			# Determine the size of the windows for the patches
			stride_height = math.floor(image_height / patches_rows)
			slide_window_height = stride_height
			stride_width = math.floor(image_width / patches_cols)
			slide_window_width = stride_width
			#print("Size: ", strideHeigth, slideWindowHeight, strideWidth, slideWindowWidth)
			# Get valid padding
			start_pixels_height = 0
			end_pixels_height = slide_window_height
			start_pixels_width = 0
			end_pixels_width = slide_window_width
			patches_coordinates = []
			numberPatches_height, numberPatches_width = get_valid_padding(slide_window_height,
																		 stride_height,
																		 image_height,
																		 slide_window_width,
																		 stride_width,
																		 image_width)
			#print('numberPatchesHeight: ', numberPatchesHeight, 'numberPatchesWidth: ', numberPatchesWidth)
<<<<<<< HEAD:ImagePreprocessing.py
			for i in range(numberPatches_height):
				for j in range(numberPatches_width):
					patches_coordinates.append([start_pixels_height,\
||||||| fbdf844... type: feature
			for i in range(number_patches_height):
				for j in range(number_patches_width):
					patches_coordinates.append([start_pixels_height,\
=======
			for i in range(number_patches_height):
				for j in range(number_patches_with):
					patchesCoordinates.append([start_pixels_height,\
>>>>>>> parent of fbdf844... type: feature:preprocess.py
													start_pixels_width,\
													end_pixels_height,\
													end_pixels_width])
					# Update width with strides
					start_pixels_width += stride_width
					end_pixels_width += stride_width
				# Re-initialize the width parameters
				start_pixels_width = 0
				end_pixels_width = stride_width
				# Update height with height stride size
				start_pixels_height += stride_height
				end_pixels_height += stride_height
			return patches_coordinates,\
					numberPatches_height,\
					numberPatches_width
		else:
			raise Exception("Type of padding not understood")

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
	:param slide_window_height: int that represents the height of the slide
								Window
	:param stride_height: int that represents the height of the stride
	:param image_height: int that represents the height of the image
	:param slide_window_width: int that represents the width of the slide
								window
	:param stride_width: int that represents the width of the stride
	:param image_width: int that represents the width of the image
	: return: a tuple containing the number of patches in the height and 
			and the width dimension.
	"""
	numberPatches_height_ = 0
	numberPatches_width_ = 0
	while(True):
		if slide_window_height <= image_height:
			slide_window_height += stride_height
			numberPatches_height_ += 1
		elif slide_window_height > image_height:
			break
		else:
			continue
	while(True):
		if slide_window_width <= image_width:
			slide_window_width += stride_width
			numberPatches_width_ += 1	
		elif slide_window_width > image_width:
			break
		else:
			continue
	return (numberPatches_height_, numberPatches_width_)

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
	:param slide_window_height: int that represents the height of the slide 
								Window
	:param stride_height: int that represents the height of the stride
	:param image_height: int that represents the height of the image
	:param slide_window_width: int that represents the width of the slide
								window
	:param stride_width: int that represents the width of the stride
	:param image_width: int that represents the width of the image
	: return: a tuple containing the amount of zeros
				to add in the height dimension and the amount of zeros
				to add in the width dimension. 
	"""
	# Initialize auxiliar variables
	numberPatches_height_ = 0
	numberPatches_width_ = 0
	# Calculate the number of patches that fit
	while(True):
		if slide_window_height <= image_height:
			slide_window_height += stride_height
			numberPatches_height_ += 1
		elif slide_window_height > image_height:
			break
		else:
			continue
	while(True):
		if slide_window_width <= image_width:
			slide_window_width += stride_width
			numberPatches_width_ += 1	
		elif slide_window_width > image_width:
			break
		else:
			continue
	# Fix the excess in slide_window
	slide_window_height -= stride_height
	slide_window_width -= stride_width
	#print(numberPatches_height_, numberPatches_width_)
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
	#print(slideWindowHeight, imageHeight, resid_h, zeros_h)
	# Return amount of zeros
	return (zeros_h, zeros_w)

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
