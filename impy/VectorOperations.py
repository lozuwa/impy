"""
Author: Rodrigo Loza
Email: lozuwaucb@gmail.com
Description: Implements mathematical operations
related to vectors.
"""
import math

class VectorOperations(object):
	def __init__(self):
		super(VectorOperations, self).__init__()

	@staticmethod
	def compute_module(vector = None):
		"""
		Computes the module of a vector.
		Args:
			vector: A list or tuple that contains the coordinates that define
							the position of the vector.
		Returns:
			A float that contains the module of the vector.
		"""
		module = math.sqrt(sum([i**2 for i in vector]))
		return module

	@staticmethod
	def euclidean_distance(v0 = None, v1 = None):
		"""
		Computes the Euler distance between two points.
		Args:
			v0: A list that contains a vector.
			v1: A list that contains another vector.
		Returns:
			An integer that contains the Euler distance.
		"""
		distance = math.sqrt(sum([(i-j)**2 for i,j in zip(v0, v1)]))
		return distance

	@staticmethod
	def rotation_equations(x, y, theta):
		"""
		Apply a 2D rotation matrix to a 2D coordinate by theta degrees.
		Args:
			x: An int that represents the x dimension of the coordinate.
			y: An int that represents the y dimension of the coordinate.
		"""
		x_result = int((x*math.cos(theta)) - (y*math.sin(theta)))
		y_result = int((x*math.sin(theta)) + (y*math.cos(theta)))
		return x_result, y_result

