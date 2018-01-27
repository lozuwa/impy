""" 
Author: Rodrigo Loza 
Company: pfm Medical Bolivia 
Description: Loads a tensorflow self.graph model
"""

# Libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import cv2
from PIL import Image
import numpy as np
from numpy import c_, r_
import tensorflow as tf
import matplotlib.pyplot as plt

class loadModel:
	"""
	Creates a class with its respective labels.txt and
	self.graph.pb for labeling
	"""

	def __init__(self,
				graph = "",
  				labels = "",
			  	input_layer = "input",
			  	output_layer = "final_result",
			  	input_mean = 128,
			  	input_std = 128,
			  	input_width = 128,
			  	input_height = 128):
		# Set model
		self.model_file = graph
		self.label_file = labels
		# Input size
		self.input_height = input_height
		self.input_width = input_width
		# Conditions
		self.input_mean = input_mean
		self.input_std = input_std
		# Tensor names
		self.input_layer = input_layer
		self.output_layer = output_layer
		# Load graph
		self.graph = self.load_graph(self.model_file)

	def predict(self,
				file_name):
		t = self.read_tensor_from_image_file(file_name,
		                                  input_height=self.input_height,
		                                  input_width=self.input_width,
		                                  input_mean=self.input_mean,
		                                  input_std=self.input_std)

		input_name = "import/" + self.input_layer
		output_name = "import/" + self.output_layer
		input_operation = self.graph.get_operation_by_name(input_name);
		output_operation = self.graph.get_operation_by_name(output_name);

		with tf.Session(graph=self.graph) as sess:
			results = sess.run(output_operation.outputs[0],
		                      {input_operation.outputs[0]: t})
			results = np.squeeze(results)
		top_k = results.argsort()[-5:][::-1]
		labels = self.load_labels(self.label_file)
		return_dict = {}
		for i in top_k:
			return_dict[labels[i]] = results[i]
			#print(labels[i], results[i])
		return return_dict

	def load_graph(self,
					model_file):
		graph = tf.Graph()
		graph_def = tf.GraphDef()
		with open(model_file, "rb") as f:
			graph_def.ParseFromString(f.read())
		with graph.as_default():
			tf.import_graph_def(graph_def)
		return graph

	def read_tensor_from_image_file(self,
									file_name,
									input_height=299,
									input_width=299,
									input_mean=0,
									input_std=255):
		input_name = "file_reader"
		output_name = "normalized"
		file_reader = tf.read_file(file_name, input_name)
		if file_name.endswith(".png"):
			image_reader = tf.image.decode_png(file_reader, channels = 3,
		                                   name='png_reader')
		elif file_name.endswith(".gif"):
			image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
		                                              name='gif_reader'))
		elif file_name.endswith(".bmp"):
			image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
		else:
			image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
		                                    name='jpeg_reader')
		float_caster = tf.cast(image_reader, tf.float32)
		dims_expander = tf.expand_dims(float_caster, 0);
		resized = tf.image.resize_bilinear(dims_expander, [self.input_height, self.input_width])
		normalized = tf.divide(tf.subtract(resized, [self.input_mean]), [self.input_std])
		sess = tf.Session()
		result = sess.run(normalized)
		return result

	def load_labels(self,
					label_file):
		label = []
		proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
		for l in proto_as_ascii_lines:
			label.append(l.rstrip())
		return label

class loadgraph:
	"""
	Creates a class with its respective labels.txt and
	self.graph.pb
	"""
	def __init__(self,
				PATH_TO_LABELS = "",
				PATH_TO_GRAPH = "" ,
				input_layer = "input:0",
				output_layer = "final_result:0",
				topK = 5):
		"""
		Args:
			PATH_TO_LABELS: input string that contains the path to output_labels.txt,
			PATH_TO_self.graph: input string that contains the path to output_self.graph.pb,
			self.input_layer: input string that contains the name of the input tensor,
			self.output_layer: input string that contains the name of the output tensor,
			topK = 5
		Returns:
			None
		Models:
			inception_v3:
				* self.input_layer = "DecodeJpeg/contents:0"
				* self.output_layer = "final_result:0"
			mobilenet:
				* self.input_layer = "input"
				* self.output_layer = "final_result"
		"""
		# Set paths
		self.PATH_TO_LABELS = os.path.join(PATH_TO_LABELS, "output_labels.txt")
		self.PATH_TO_graph = os.path.join(PATH_TO_self.graph, "output_self.graph.pb")
		# Set layers
		self.self.input_layer = self.input_layer
		self.self.output_layer = self.output_layer
		# Configuration
		self.topK = topK
		self.label_lines =  [line.rstrip() for line in\
							tf.gfile.GFile(self.PATH_TO_LABELS)]
		with tf.gfile.FastGFile(self.PATH_TO_GRAPH, "rb") as f:
			self.graph_def = tf.graphDef()
			self.graph_def.ParseFromString(f.read())
			self.__ = tf.import_self.graph_def(self.graph_def,\
											name="")
		self.sess = tf.Session()
		self.softmax_tensor = self.sess.self.graph.get_tensor_by_name(self.output_layer)

	def predict(self,
				image_path):
		"""
		Infers the class of a single image
		Args:
			image_path: str that indicates the path of image
		Returns: 
			a dictionary that contains the classes and the
			probabilities
		"""
		image_data = tf.gfile.FastGFile(image_path, "rb").read()
		predictions = self.sess.run(self.softmax_tensor, {self.self.input_layer: image_data})
		predictions_ = predictions[0].argsort()[-len(predictions[0]):][::-1]
		return_dict = {}
		for node_id in predictions_[:self.topK]:
			human_string = self.label_lines[node_id]
			score = predictions[0][node_id]
			return_dict[human_string] = score
		return return_dict
