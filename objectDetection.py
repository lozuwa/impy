# General purpose
import os
import sys
import six.moves.urllib as urllib
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
import cv2

# Object detection libraries
sys.path.append("/home/pfm/Documents/models/research/object_detection/")
from utils import label_map_util
from utils import visualization_utils as vis_util

# Impy
from impy.preprocess import *

class objectDetection():
	def __init__(self):
		# Path to model
		MODEL_NAME = ["rcnn_resnet_101_v1", "ssd_inception_v2"]
		PATH_TO_CKPT = os.path.join("/home/pfm/Documents/models/research/data/train", MODEL_NAME[0], "frozen_inference_graph.pb")
		# List of the strings that is used to add correct label for each box.
		PATH_TO_LABELS = os.path.join("/home/pfm/Documents/models/research/data", "label_map.pbtxt")
		assert os.path.isfile(PATH_TO_CKPT), "path to frozen graph is not working"
		assert os.path.isfile(PATH_TO_LABELS), "path to labels is not working"
		#asssert os.path.isfile(PATH_TO_CKPT), "frozen_inference_graph path is invalid"
		#asssert os.path.isfile(PATH_TO_LABELS), "label_map.pbtxt path is invalid"
		NUM_CLASSES = 1

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
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)
		assert os.path.isdir(folder_directory), "folder does not exist"
		dir_, folder_ = os.path.split(folder_directory)
		imgs = [os.path.join(folder_directory, each) for each in os.listdir(folder_directory) if each.endswith(".jpg")]
		# Size, in inches, of the output images.
		IMAGE_SIZE = (12, 8)
		pi = preprocessImage()
		# Run inference
		with detection_graph.as_default():
			with tf.Session(graph = detection_graph) as sess:
				for img in imgs:
					# Preprocess the image
					frame = cv2.imread(img)
					height, width, depth = frame.shape
					slideWindowSize = (1050, 1050)
					strideSize = (1050, 1050)
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
						image_path = os.path.join(os.getcwd(), "image"+str(counter)+".jpg")
						cv2.imwrite(image_path, frame[iw:ew, ih:eh, :])
						counter += 1
					
					# Inference
					# Definite input and output Tensors for detection_graph
					image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
					# Each box represents a part of the image where a particular object was detected.
					detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
					# Each score represent how level of confidence for each of the objects.
					# Score is shown on the result image, together with the class label.
					detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
					detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
					num_detections = detection_graph.get_tensor_by_name("num_detections:0")
					PROCESSED_IMAGE_PATHS = [os.path.join("/home/pfm/Documents/models/research/object_detection/", "image{}.jpg".format(str(each))) for each in range(counter)]
					print(PROCESSED_IMAGE_PATHS)
					for image_path in PROCESSED_IMAGE_PATHS:
						image = Image.open(image_path)
						# the array based representation of the image will be used later in order 
						# to prepare the result image with boxes and labels on it.
						image_np = load_image_into_numpy_array(image)
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
																			category_index,
																			use_normalized_coordinates=True,
																			line_thickness = 8)
					plt.figure(figsize=IMAGE_SIZE)
					plt.imshow(image_np)
