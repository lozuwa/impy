"""
package: impy
class: Images2Dataset
Author: Rodrigo Loza
Description: Class that loads an image dataset.
We recommend giving some structure to your dataset before
using this class.
* The folder's structure should be like this:
						- Data set/
								- Folder class 0/
									- Image 1
									- Image 2 
									- ...
								- Folder class 1/
									- Image 1
									- Image 2 
									- ...
								- ...
- You should provide the path to the Data set folder.
- We recommend giving the data set folder a descriptive name.
- The nested folders should contain the name of the class.
- The file names of the images inside the folders don't matter. If you
	would like to give them proper names, then use the formatDataset method.
"""
# Utils
import os
import datetime

class Images2Dataset(object):
		def __repr__(self):
			return "Images2Dataset({})".format("PathToSomeFolder")

		def __init__(self,
									dbFolder = None,
									datasetFormat = None):
			"""
			Constructor.
			"""
			if dbFolder == None:
				raise ValueError("You must provide a folder.")
			if datasetFormat == None:
				datasetFormat = "normal"
			# Assert paths
			if not os.path.isdir(dbFolder):
				raise Exception("Path to foler does not exist.")
			# Save the path to dbFolder
			self.pathDbFolder = dbFolder
			# Create a dictionary with the folders and its images
			self.foldersAndFiles = self.getFolderAndFiles(parentFolder = dbFolder)

		@property
		def propertyfoldersAndFiles(self):
			return self.foldersAndFiles

		@property.setter
		def propertyfoldersAndFiles(self,
																	newFoldersAndFiles):
			self.foldersAndFiles = newFoldersAndFiles

		def getFolderAndFiles(self, parentFolder = None):
			"""
			Get a data structure with the folders and the files.
			Args:
				parentFolder: A list containing folder names.
			Returns:
				A hashmap containing the folders and its images. 
				{"folder1": [image1, image2, ...], "folder2": [image1, image2, ...]}
			"""
			# Local vairables
			if parentFolder == None:
				raise ValueError("ParentFolder cannot be empty.")
			foldersAndFiles = {}
			# Read folders
			foldersInParent = os.listdir(parentFolder)
			for folderInParent in foldersInParent:
				files = os.listdir(os.path.join(parentFolder, folderInParent))
				files = self.filterAllowedImageFormats(files)
				foldersAndFiles[folderInParent] = files
			return foldersAndFiles

		def filterAllowedImageFormats(self, files):
			"""
			Filter Image files.
			Args:
				files: A list containing image files.
			Returns:
				A list containing the filetered files with the 
				allowed formats.
			"""
			ALLOWED_FILE_FORMATS = ["jpg", "png"]
			allowedFiles = []
			for f in files:
				if f.split(".")[1] in ALLOWED_FILE_FORMATS:
					allowedFiles.append(f)
			return allowedFiles

		def formatDataset(self):
			# Local variables
			newFoldersAndFiles = {}
			# Iterate over class names
			for className in self.foldersAndFiles.keys():
				# Update className
				newFoldersAndFiles[className] = []
				# Get files names
				fileNames = self.foldersAndFiles[className]
				# Iterate over file names
				for fileName in fileNames:
					# Get path to fileName
					pathToFileName = os.path.join(self.pathDbFolder,
																				className,
																				fileName)
					# Assert path to file
					if not os.path.isfile(pathToFileName):
						raise Exception("Path to file does not exist.")
					# Create a new name for the file
					newFileName = "_".join([className, now()]) + ".jpg"
					newPathToFileName = os.path.join(os.path.split(pathToFileName)[0],
																						newFileName)
					# Update filename
					newFoldersAndFiles[className].append(newFileName)
					# Rename file with its respective class name
					os.rename(pathToFileName, newPathToFileName)
			# Set dictionary again
			self.foldersAndFiles = newFoldersAndFiles

		def computeStats(self):
			print("Distribution of classes and data points: ")
			for folder, files in zip(self.foldersAndFiles.keys(),
																self.foldersAndFiles.values()):
				print("{}: {}".format(folder, len(files)))

def now():
	"""
	Staticmehtod that returns a date.
	Returns:
		A string that contains the actual time in the following format:
			day_minute_second_microsecond.
	"""
	now = datetime.datetime.now()
	now = list(map(str, [now.day, now.minute,
												now.second, now.microsecond]))
	return "{}_{}_{}_{}".format(now[0], now[1], now[2], now[3])
		