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
# Utils
from utils import *
# Visualization
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

class EDA:
	def __init__(self, data = {}):
		""" 
		Exploratory data analysis class
		"""
		self.data = data

	def classesBalance(self, visualize = False):
		"""
		:param visualize: bool that decides if output visualization
		: return: a list that depicts each class absolute frequency 
		"""
		classes = getDictKeys(self.data)
		values = []
		for class_ in classes:
			absoluteFrequency = len(getDictValues(self.data, class_))
			values.append(absoluteFrequency)
		if visualize:
			self.scatterPlot(classes, values, title = "Class balance", xlabel = "x", ylabel = "Frequencies")
		else:
			return classes, values

	def scatterPlot(self, x = [0], y = [0], title = "ScatterPlot", xlabel = "x", ylabel = "y"):
		"""
		:param x: a list containing the x values 
		:param y: a list containing the y values
		:param title: a string containing the name of the plot
		:param xlabel: a string containing the xlabel name
		:param ylabel: a string containing the ylabel name
		: return: Scatter plot 
		"""
		print(x)
		x = [i for i in range(len(x))]
		plt.scatter(x, y)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	def getMean(self, totalPixels = 1):
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
