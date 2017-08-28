"""
package: Images2Dataset
class: stats
Author: Rodrigo Loza
Description: Utils methods 
"""
# General purpose 
import os
import sys
# Image manipulation
import cv2
from PIL import Image
# Visualization
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
# Collections
from collections import Counter
# Tensor manipulation
import numpy as np
# Utils
from .utils import *

class stats:
	def __init__(self, 
				data = {}):
		""" 
		Exploratory data analysis class
		"""
		self.data = data

	def classesBalance(self, 
						visualize = False):
		"""
		:param visualize: bool that decides if output visualization
		: return: a list that depicts each class absolute frequency 
		"""
		classes = getDictKeys(self.data)
		values = []
		for class_ in classes:
			vals = [each for each in getDictValues(self.data, class_)]
			# Filter nans
			for i in range(vals.count(np.nan)):
				vals.remove(np.nan)
			# Get absolute frequency
			absoluteFrequency = len(vals)
			values.append(absoluteFrequency)
		if visualize:
			sendClass = []
			# Clean xticks names
			for class_ in classes:
				sendClass.append(class_.split("/")[-1])
			# Plot 
			self.scatterPlot(x = sendClass,
							y = values, 
							xticks = True,
							title = "Class balance", 
							xlabel = "x", 
							ylabel = "Frequencies")
		else:
			return classes, values

	def scatterPlot(self, 
					x = [0], 
					y = [0], 
					xticks = False,
					title = "ScatterPlot", 
					xlabel = "x", 
					ylabel = "y"):
		"""
		:param x: a list containing the x values 
		:param y: a list containing the y values
		:param title: a string containing the name of the plot
		:param xlabel: a string containing the xlabel name
		:param ylabel: a string containing the ylabel name
		: return: Scatter plot 
		"""
		x_ = [i for i in range(len(x))]
		if xticks:
			my_xticks = x
			assert len(my_xticks) == len(x_),\
					VECTORS_MUST_BE_OF_EQUAL_SHAPE 
			plt.xticks(x_, my_xticks)
		else:
			pass	
		plt.scatter(x_, y)
		plt.title(title)
		plt.ylim([0, max(y)])
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	def tensorSizes(self):
		"""
		Computes the most frequent sizes in the dataset 
		"""
		# Variables
		sizes = []
		depths = []
		# Iterate over images 
		keys = getDictKeys(self.data)
		for key in keys:
			imgs = self.data[key]
			for img in imgs:
				if type(img) == str:
					frame = Image.open(img)
					sizes.append(frame.size)
					depths.append(len(frame.getbands()))
				else:
					pass
					#raise ValueError("Data type not understood {}".format(type(img)))
		frame.close()
		s = Counter(sizes)
		d = Counter(depths)
		print("***Size tensor")
		print("All sizes: ", set(sizes))
		print("All depths: ", set(depths))
		print("Most common size: ", s.most_common)
		print("Most common depths: ", d.most_common)

	def getMean(self, 
				totalPixels = 1):
		"""
		Calculates the mean of the image dataset
		using the .jpg/.png/.jpeg files. 
		"""
		# Get keys
		keys = getDictKeys(self.data)
		# Read images in each subfolder
		for k in tqdm(range(len(keys))):
			# Get images of class "key"
			assert keys[k] != None, "There was a problem with key"
			imgs = self.images.get(keys[k])
			# Read image
			for img in tqdm(imgs):
				# Read image and scale to 0-1
				numerator += np.sum(cv2.imread(img).reshape(-1, 1) / 255)

		# Calculate mean and standard deviation
		mean = numerator / (totalPixels)

		return mean
