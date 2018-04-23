import os
import json
from AssertJsonConfiguration import *
jsonConf = AssertJsonConfiguration()

def main():
	# Load augmentation file
	file = os.path.join(os.getcwd(), "tests", "augmentation.json")
	data = json.load(open(file))
	jsonConf.
	# Iterate over all images
	for img in range(1):
		# Read all image information
		img = os.path.join(os.getcwd())
		# Load image and annotation 
		# ...
		# Get bounding box augmenters for data augmentation
		for i in data["bounding_box_augmenters"]:
			if (i == "Sequential"):
				for j in data["bounding_box_augmenters"][i]:
					parameters = data["bounding_box_augmenters"][i][j]
					frame, bndboxes = __applyBoundingBoxAugmentation__(frame = None,
																														boundingBoxes = None,
																														names = None,
																														augmentation_type = j, 
																														parameters = parameters,
																														returnInfo = True)
			else:
				parameters = data["bounding_box_augmenters"][i]
				__applyBoundingBoxAugmentation__(frame = None, # cv2.imread(ImagePath)
																					boundingBoxes = None,
																					names = None,
																					augmentation_type = i,
																					parameters = parameters,
																					returnInfo = False)

def __applyBoundingBoxAugmentation__(frame = None,
																			boundingBoxes = None,
																			names = None,
																			augmentation_type = None,
																			parameters = None,
																			returnInfo = None):
	if augmentation_type == "scale":
		frame = 1
		bndboxes = 1
		print("Scale: ", parameters)
	elif augmentation_type == "crop":
		frame = 2
		bndboxes = 2
		print("Crop: ", parameters)
	elif augmentation_type == "pad":
		frame = 3
		bndboxes = 3
		print("Pad: ", parameters)
	elif augmentation_type == "jitterBoxes":
		frame = 4
		bndboxes = 4
		print("jitterBoxes: ", parameters)
	else:
		pass
	if returnInfo:
		return frame, bndboxes
	else:
		pass	
				
if __name__ == "__main__":
	main()

