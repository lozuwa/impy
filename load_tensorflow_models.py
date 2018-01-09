""" Author: Rodrigo Loza 
 Company: pfm Bolivia 
 Description: Script that is used to load a graph for prediction
"""

# Libraries
import os
import sys
import platform, cv2
from PIL import Image
import numpy as np
from numpy import c_, r_
import tensorflow as tf

class loadGraph:
	"""
	Creates a class with its respective labels.txt and
	graph.pb
	"""
	def __init__(self,
				PATH_GRAPH = "",
				PATH_LABELS = "",
				verbosity = False,
				topK = 5):
		"""
		:param verbosity: boolean that decides the
							verbosity in the prediction
		"""
		# Set paths
		self.PATH_TO_LABELS = INCEPTION_GOOGLE_EXAMPLE_GRAPH + "output_labels.txt"
		self.PATH_TO_GRAPH = INCEPTION_GOOGLE_EXAMPLE_GRAPH + "output_graph.pb"
		# Set options
		self.verbosity = verbosity
		self.topK = topK
		# Process graph and labels
		self.label_lines =  [line.rstrip() for line in\
							tf.gfile.GFile(PATH_TO_LABELS)]
		with tf.gfile.FastGFile(PATH_TO_GRAPH, "rb") as f:
			self.graph_def = tf.GraphDef()
			self.graph_def.ParseFromString(f.read())
			self.__ = tf.import_graph_def(self.graph_def,\
											name="")
		self.sess = tf.Session()
		self.softmax_tensor = self.sess.graph.get_tensor_by_name("final_result:0")
		# Create impy preprocess instance
		self.pi = preprocessImage()

	def predict(self,
				image_path):
		"""
		Infers the class of a single image
		:param image_path: str that indicates the path of image
		: return: a dictionary that contains the classes and the
					probabilities
		"""
		image_data = tf.gfile.FastGFile(image_path, "rb").read()
		predictions = self.sess.run(self.softmax_tensor, {"DecodeJpeg/contents:0": image_data})
		predictions_ = predictions[0].argsort()[-len(predictions[0]):][::-1]
		return_dict = {}
		for node_id in predictions_[:self.topK]:
			human_string = self.label_lines[node_id]
			score = predictions[0][node_id]
			return_dict[human_string] = score
			if self.verbosity:
				print("%s (score = %.5f)" % (human_string, score))
		return return_dict
