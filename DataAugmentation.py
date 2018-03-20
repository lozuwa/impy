"""
package: Images2Dataset
class: DataAugmentation
Author: Rodrigo Loza
Description: Common data augmentation operations 
for an image.
"""
# Libraries
from interface import Implements

class DataAugmentation(Implements(DataAugmentationMethods)):
	"""
	DataAugmentation class.
	"""
	def __init__(self, frame):
		# Super class
		super(DataAugmentation, self).__init__()
		# Create a session
		self.sess = tf.Session()

	def centralCropping(self, frame):
		"""
		Central crops an image using the tensorflow embedded function.
		"""
		central_fraction = np.random.rand()
		frame = tf.image.central_crop(frame, central_fraction)
		return frame

	def cropAndResize(self, frame):
		"""
		Crop and image and resize it.
		"""
		frame = tf.image.resize_image_with_crop_or_pad(frame,
																									image,
																									boxes,
																									box_ind,
																									crop_size,
																									method='bilinear')
		return frame

	def jitterBox(self, frame):
		# Assert dimensions
		assert len(frame.shape) == 3, "Image has to be 3 dimensional"
		# Compute properties
		height, width, depth = frame.shape
		#

	def flipHorizontalOrVertical(self, frame):
		choice = bool(int(np.random.rand()*2))
		if choice:
			frame = tf.flip_left_right(frame)
		else:
			frame = tf.flip_up_down(frame)
		return frame
