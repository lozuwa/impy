"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: A class that performs processing operations with image annotations.
"""
# General purpose
import os 
import xml.etree.ElementTree as ET
import numpy as np

class AnnotationProcessing(object):
	def __init__(self):
		super(AnnotationProcessing, self).__init__()
	
	def iou(self, bndbx1 = None, bndbx2 = None):
		"""
		A method that is used to compute the intersection over union (IoU) between two
		bounding boxes.
		IoU = Area of overlap / area of union
		Args:
			bndbx1: A list of ints that contains the coordinates of a bounding box.
			bndbx2: A list of ints that contains the coordinates of a bounding box.
		Returns:
			A float that contains the intersection over union.
		"""
		# Assertions.
		if (bndbx1 == None):
			raise TypeError("Bounding box 1 cannot be empty.")
		if (bndbx2 == None):
			raise TypeError("Bounding box 2 cannot be empty.")
		# Local variables.
		ix1, iy1, x1, y1 = bndbx1
		ix2, iy2, x2, y2 = bndbx2
		# Logic.
		# Find the intersection.
		intIx = max(ix1, ix2)
		intIy = max(iy1, iy2)
		intX = min(x1, x2)
		intY = min(y1, y2)
		areaOverlap = (intY - intIy + 1) * (intX - intIx + 1)
		# Area of each bounding box.
		areaBndbx1 = (y1 - iy1 + 1) * (x1 - ix1 + 1)
		areaBndbx2 = (y2 - iy2 + 1) * (x2- ix2 + 1)		
		# Find the area of union.
		areaUnion = (areaBndbx1 + areaBndbx2) - areaOverlap
		# IoU
		iou = areaOverlap / areaUnion
		return iou

	def nonMaxSuppression(self, boxes = None, overlapThresh = None):
		"""
		Given a list of bounding boxes, find the region that best includes the object
		given an overlap threshold.
		Args:
			boxes: A list of lists of ints that contains bounding boxes.
			overlapThresh: A float in the range [0, 1].
		Returns:
			A single bounding box that contains an object.
		"""
		# Assertions.
		if (boxes == None):
			raise Exception("Boxes cannot be empty.")
		if (overlapThresh == None):
			raise Exception("Overlap threshold cannot be empty.")
		if (type(boxes) == list):
			boxes = np.array(boxes)
		if (type(boxes) != np.ndarray):
			raise TypeError("Boxes must be a list or a numpy array.")
		if (len(boxes) == 0):
			raise Exception("Boxes cannot be empty.")
		# Local variables.
		pick = []	 
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]
		# Logic.
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")	 
		# Compute the area of the bounding boxes and sort the bounding
		# boxes by the bottom-right y-coordinate of the bounding box.
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
		# Keep looping while some indexes still remain in the indexes list.
		while len(idxs) > 0:
			# Grab the last index in the indexes list and add the
			# index value to the list of picked indexes.
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
			# Find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box.
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])
			# Compute the width and height of the bounding box.
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
			# Compute the ratio of overlap.
			overlap = (w * h) / area[idxs[:last]]
			# Delete all indexes from the index list that have.
			idxs = np.delete(idxs, np.concatenate(([last],
				np.where(overlap > overlapThresh)[0])))
		# Return only the bounding boxes that were picked using the
		# integer data type.
		if (len(pick) != 1):
			pick = [pick[0]]
		return boxes[pick].astype("int")


