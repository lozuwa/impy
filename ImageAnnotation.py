"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Decomposes the information contained in an image 
annotation with the VOC format.
"""
import os
from interface import implements
import xml.etree.ElementTree as ET

class ImageAnnotation(object):
	def __init__(self, path):
		super(ImageAnnotation, self).__init__()
		self.path = path
		self.root = self.readImageAnnotation(self.path)
		self.size = self.getSize(self.root)
		self.objects = self.getObjects(self.root)
		self.names = self.getNames(self.objects)
		self.boundingBoxes = self.getBoundingBoxes(self.objects)

	@property
	def propertySize(self):
		return self.size

	@property
	def propertyObjects(self):
		return self.objects

	@property
	def propertyNames(self):
		return self.names

	@property
	def propertyBoundingBoxes(self):
		return self.boundingBoxes

	def readImageAnnotation(self, 
													path):
		tree = ET.parse(path)
		root = tree.getroot()
		return root

	def getObjects(self, 
								root):
		if (root.find("object")):
			objects = root.findall("object")
			return objects
		else:
			print("WARNING: No objects found.")
			return None

	def getNames(self, 
								objects):
		names = []
		for obj in objects:
			names.append(obj.find("name").text)
		return names

	def getBoundingBoxes(self, 
												objects):
		boundingBoxes = []
		for i in range(len(objects)):
			# Find bndbox
			coordinates = objects[i].find("bndbox")
			# Get coordinates
			xmin = int(coordinates.find("xmin").text)
			xmax = int(coordinates.find("xmax").text)
			ymin = int(coordinates.find("ymin").text)
			ymax = int(coordinates.find("ymax").text)
			boundingBoxes.append([xmin, ymin, xmax, ymax])
		return boundingBoxes

	def getSize(self, root):
		if (root.find("size")):
			size = root.find("size")
			width = int(size[0].text)
			height = int(size[1].text)
			depth = int(size[2].text)
			return [width, height, depth]
		else:
			return None
