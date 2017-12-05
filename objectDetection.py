# General purpose
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO

# Tensor manipulation
import numpy as np
import tensorflow as tf

# Image manipulation and visualization
from matplotlib import pyplot as plt
from PIL import Image

# Object detection libraries
from utils import label_map_util
from utils import visualization_utils as vis_util

class objectDetection(object):
	def __init__(self):
		# Path to model
		self.MODEL_NAME = "ssd_mobilenet_v1_coco_2017_11_17"
		self.PATH_TO_CKPT = os.path.join(os.getcwd(), MODEL_NAME, "/frozen_inference_graph.pb")
		self.PATH_TO_LABELS = os.path.join(os.getcwd(), "data", "label_map.pbtxt")
		self.NUM_CLASSES = 1
		# Load frozen graph
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			self.od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
			serialized_graph = fid.read()
			self.od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(self.od_graph_def, name="")
		# Load labels
		self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
																	max_num_classes = self.NUM_CLASSES,
																	use_display_name = True)
		self.category_index = label_map_util.create_category_index(self.categories)

	def load_image_into_numpy_array(self,
									image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	def classify(self,
				folder_directory):
		"""
		Compute inference on the images of a folder
		:param folder_directory: input string that contains the name of a folder
		"""
		dir_, folder_ = os.path.split(folder_directory)
		imgs = [os.path.join(folder_directory, each) for each in os.listdir(folder_directory) if each.endswith(".jpg")]
		# Size, in inches, of the output images.
		IMAGE_SIZE = (12, 8)
		# Run inference
		with self.detection_graph.as_default():
			with tf.Session(graph = self.detection_graph) as sess:
				for img in imgs:
					# Preprocess the image
					frame = cv2.imread(img)
					height, width, depth = frame.shape
					slideWindowSize = (1050, 1050)
					strideSize = (525, 525)
					padding = "SAME"
					patches, numberPatchesHeight, numberPatchesWidth,\
					zeros_h, zeros_w = pi.divideIntoPatches(width,
										                    height,
										                    slideWindowSize,
										                    strideSize,
										                    padding)
					frame = lazySAMEpad(frame.copy(),
					                    zeros_h,
					                    zeros_w)
					counter = 0
					for patch in patches:
						print(counter)
						iw, ih, ew, eh = patch
						cv2.imwrite("image"+str(counter)+".jpg", frame[iw:ew, ih:eh, :])
						counter += 1
					# Inference
					# Definite input and output Tensors for detection_graph
					image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
					# Each box represents a part of the image where a particular object was detected.
					detection_boxes = self.detection_graph.get_tensor_by_name("detection_boxes:0")
					# Each score represent how level of confidence for each of the objects.
					# Score is shown on the result image, together with the class label.
					detection_scores = self.detection_graph.get_tensor_by_name("detection_scores:0")
					detection_classes = self.detection_graph.get_tensor_by_name("detection_classes:0")
					num_detections = self.detection_graph.get_tensor_by_name("num_detections:0")
					PROCESSED_IMAGE_PATHS = [os.path.join(folder_directory, each) for each in range(counter)]
					for image_path in PROCESSED_IMAGE_PATHS:
						image = Image.open(image_path)
						# the array based representation of the image will be used later in order 
						# to prepare the result image with boxes and labels on it.
						image_np = self.load_image_into_numpy_array(image)
						# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
						image_np_expanded = np.expand_dims(image_np, axis=0)
						# Actual detection
						(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
																feed_dict={image_tensor: image_np_expanded})
						# Visualization of the results of a detection.
						vis_util.visualize_boxes_and_labels_on_image_array(image_np,
																			np.squeeze(boxes),
																			np.squeeze(classes).astype(np.int32),
																			np.squeeze(scores),
																			self.category_index,
																			use_normalized_coordinates=True,
																			line_thickness = 8)
					plt.figure(figsize=IMAGE_SIZE)
					plt.imshow(image_np)
