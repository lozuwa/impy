class PreprocessImageDataset:
	"""
	Allows operations on images
	"""
	def __init__(self,
				data = []):
		"""
		Constructor of the preprcoessImageDataset
		:param data: pandas dataframe that contains as features the
					name of the classes and as values the paths of
					the images.
		"""
		self.data = data

	def resizeImages(self,
					width = 300,
					height = 300):
		"""
		Resizes all the images in the db. The new images are stored
		in a local folder named "dbResized"
		:param width: integer that tells the width to resize
		:param height: integer that tells the height to resize
		"""
		# Create new directory
		DB_PATH = os.path.join(os.getcwd(), "dbResized")
		assert createFolder(DB_PATH) == True,\
				PROBLEM_CREATING_FOLDER
		# Read Images
		keys = getDictKeys(self.data)
		for key in tqdm(keys):
			imgs = self.data.get(key, None)
			# Create subfolders for each subfolder
			DIR, NAME_SUBFOLDER = os.path.split(key)
			#NAME_SUBFOLDER = key.split("/")[-1]
			assert createFolder(os.path.join(DB_PATH, NAME_SUBFOLDER)) == True,\
					PROBLEM_CREATING_FOLDER
			# Iterate images
			for img in imgs:
				# Filter nan values
				if type(img) == str:
					# Open image
					frame = Image.open(img)
					# Resize image
					frame = frame.resize((width, height),\
											PIL.Image.LANCZOS)
					dir_, IMAGE_NAME = os.path.split(img)
					#IMAGE_NAME = "/" + img.split("/")[-1]
					# Save the image
					frame.save(os.path.join(DB_PATH, NAME_SUBFOLDER, IMAGE_NAME))
				else:
					pass
		print(RESIZING_COMPLETE)

	def rgb2gray(self):
		"""
		Converts the images to grayscale. The new images are stored
		in a local folder named dbGray
		"""
		# Create new directory
		DB_PATH = os.getcwd() + "/dbGray/"
		assert createFolder(DB_PATH) == True,\
				PROBLEM_CREATING_FOLDER
		# Read images
		keys = getDictKeys(self.data)
		for key in tqdm(keys):
			imgs = self.data.get(key, None)
			# Create subfolders
			NAME_SUBFOLDER = key.split("/")[-1]
			assert createFolder(DB_PATH + NAME_SUBFOLDER) == True,\
					PROBLEM_CREATING_FOLDER
			for img in imgs:
				# Filter nan values
				if type(img) == str:
					# Read image
					frame = Image.open(img)
					# Convert RGBA to GRAYSCALE
					frame = frame.convert(mode = "1") #dither = PIL.FLOYDSTEINBERG)
					# Save the image
					IMAGE_NAME = "/" + img.split("/")[-1]
					frame.save(DB_PATH + NAME_SUBFOLDER + IMAGE_NAME)
				else:
					pass
		print(RBG2GRAY_COMPLETE)

	def splitImageDataset(self,
						trainSize = 0.80,
						validationSize = 0.20):
		"""
		Splits the dataset into a training and a validation set
		:param trainSize: int that represents the size of the training set
		:param validationSize: int that represents the size of the
								validation set
		: return: four vectors that contain the training and test examples
					trainImgsPaths, trainImgsClass
					testImgsPaths, testImgsClass
		"""
		# Split dataset in the way that we get the paths of the images
		# in a train set and a test set.
		# ----- | class0, class1, class2, class3, class4
		# TRAIN | path0   path1   path2   path3   path4
		# TRAIN | path0   path1   path2   path3   path4
		# TRAIN | path0   path1   path2   path3   path4
		# TRAIN | path0   path1   path2   path3   path4
		# TEST* | path0   path1   path2   path3   path4
		# TEST* | path0   path1   path2   path3   path4
		trainX, testX, trainY, testY = train_test_split(self.data,\
														self.data,\
														train_size = trainSize,\
														test_size = validationSize)
		print(trainX.shape, testX.shape)
		# Once the dataset has been partitioned, we are going
		# append the paths of the matrix into a single vector.
		# TRAIN will hold all the classes' paths in train
		# TEST will hold all the classes' paths in test
		# Each class wil have the same amount of examples.
		trainImgsPaths = []
		trainImgsClass = []
		testImgsPaths = []
		testImgsClass = []
		# Append TRAIN and TEST images
		keys = self.data.keys()
		for key in keys:
			imgsTrain = trainX[key]
			imgsTest = testX[key]
			for imgTrain in imgsTrain:
				if type(imgTrain) != str:
					pass
				else:
					trainImgsPaths.append(imgTrain)
					trainImgsClass.append(key)
			for imgTest in imgsTest:
				if type(imgTest) != str:
					pass
				else:
					testImgsPaths.append(imgTest)
					testImgsClass.append(key)
		return trainImgsPaths,\
				trainImgsClass,\
				testImgsPaths,\
				testImgsClass

	def saveImageDatasetKeras(self,
								trainImgsPaths,
								trainImgsClass,
								testImgsPaths,
								testImgsClass):
		# Create folder
		DB_PATH = os.getcwd() + "/dbKerasFormat/"
		result = createFolder(DB_PATH)
		assert result == True,\
				PROBLEM_CREATING_FOLDER
		# Create train subfolder
		TRAIN_SUBFOLDER = "train/"
		result = createFolder(DB_PATH + TRAIN_SUBFOLDER)
		assert result == True,\
				PROBLEM_CREATING_FOLDER
		# Create validation subfolder
		VALIDATION_SUBFOLDER = "validation/"
		result = createFolder(DB_PATH + VALIDATION_SUBFOLDER)
		assert result == True,\
				PROBLEM_CREATING_FOLDER
		# Create classes folders inside train and validation
		keys = self.data.keys()
		for key in keys:
			# Train subfolder
			NAME_SUBFOLDER = key.split("/")[-1]
			result = createFolder(DB_PATH + TRAIN_SUBFOLDER +\
								 NAME_SUBFOLDER)
			assert result == True,\
					PROBLEM_CREATING_FOLDER
			# Test subfolder
			NAME_SUBFOLDER = key.split("/")[-1]
			result = createFolder(DB_PATH + VALIDATION_SUBFOLDER +\
								 NAME_SUBFOLDER)
			assert result == True,\
					PROBLEM_CREATING_FOLDER

		######################## OPTIMIZE ########################
		# Save train images
		# Read classes in trainImgsClass
		for i in tqdm(range(len(trainImgsClass))):
			imgClass = trainImgsClass[i].split("/")[-1]
			for key in keys:
				NAME_SUBFOLDER = key.split("/")[-1]
				#print(imgClass, NAME_SUBFOLDER)
				# If they are the same class, then save the image
				if imgClass == NAME_SUBFOLDER:
					NAME_SUBFOLDER += "/"
					NAME_IMG = trainImgsPaths[i]
					frame = Image.open(NAME_IMG)
					NAME_IMG = NAME_IMG.split("/")[-1]
					frame.save(DB_PATH + TRAIN_SUBFOLDER +\
								NAME_SUBFOLDER + NAME_IMG)
				else:
					pass
		# Save test images
		# Read classes in testImgsClass
		for i in tqdm(range(len(testImgsClass))):
			imgClass = testImgsClass[i].split("/")[-1]
			for key in keys: 
				NAME_SUBFOLDER = key.split("/")[-1]
				# If they are the same class, then save the image
				if imgClass == NAME_SUBFOLDER:
					NAME_SUBFOLDER += "/"
					NAME_IMG = testImgsPaths[i]
					frame = Image.open(NAME_IMG)
					NAME_IMG = NAME_IMG.split("/")[-1]
					frame.save(DB_PATH + VALIDATION_SUBFOLDER +\
								NAME_SUBFOLDER + NAME_IMG)
				else:
					pass
		##########################################################


	def resizeImage(self,
									frame,
									height = None,
									width = None):
		"""
		Args:
			frame: An opencv image.
			height: An int that contains how much to resize an image for the 
							height axis.
			width: An int that contains how much to resize an image for the 
							width axis.
		Return:
			An opencv image resized with the given parameters.
		"""
		if height == None:
			height = 300
		if width == None:
			width = 300
		resized_frame = cv2.resize(frame, (height, width))
		frame = None
		return resized_frame

	def transformGray(self,
										frame):
		"""
		Transform an image to grayscale.
		Args:
			frame: An opencv image.
		Returns:
			An opencv image in grayscale color format.
		"""
		assert len(frame.shape) == 3, "Image is not 3 dimensional," + \
																		"cannot convert to gray."
		frame_gray = cv2.cvtColor(cv2.CV_COLOR_RGB2GRAY)
		return frame_gray



def VOCFormat(self,
										folderPathAnnotations,
										folder,
										filename,
										path,
										database,
										width,
										height,
										depth,
										name,
										xmin,
										xmax,
										ymin,
										ymax):
				"""
				Method that converts an image to a xml annotation format in the 
				PASCAL VOC dataset format. 
				:param folderPath: input string with the path of the folder
				:param folder: input string with the name "annotations"
				:param path: input string with the path of the image
				:param database: input string with the name of the database
				:param width: input int with the width of the image
				:param height: input int with the height of the image
				:param depth: input int with the depth of the image
				:param name: input string with the name of the class the image
										belongs to
				:param xmin: input int with the pixel the cropping starts in the
										width of the image
				:param xmax: input int with the pixel the cropping ends in the
										width of the image
				:param ymin: input int with the pixel the cropping starts in the 
										height of the image
				:param ymax: input int with the pixel the cropping ends in the 
										height of the image
				"""
				# Create annotation header
				annotation = ET.Element("annotation")
				# Image information
				ET.SubElement(annotation, "folder", verified = "yes").text = str(folder)
				ET.SubElement(annotation, "filename").text = str(filename)
				ET.SubElement(annotation, "path").text = str(path)
				# Source
				source = ET.SubElement(annotation, "source")
				ET.SubElement(source, "database").text = str(database)
				# Size
				size = ET.SubElement(annotation, "size")
				ET.SubElement(size, "width").text = str(width)
				ET.SubElement(size, "height").text = str(height)
				ET.SubElement(size, "depth").text = str(depth)
				# Segemented
				ET.SubElement(annotation, "segmented").text = "0"
				# Object
				object_ = ET.SubElement(annotation, "object")
				ET.SubElement(object_, "name").text = str(name)
				ET.SubElement(object_, "pose").text = "Unspecified"
				ET.SubElement(object_, "truncated").text = "0"
				ET.SubElement(object_, "difficult").text = "0"
				# Bound box inside object
				bndbox = ET.SubElement(object_, "bndbox")
				ET.SubElement(bndbox, "xmin").text = str(xmin)
				ET.SubElement(bndbox, "xmax").text = str(xmax)
				ET.SubElement(bndbox, "ymin").text = str(ymin)
				ET.SubElement(bndbox, "ymax").text = str(ymax)
				# Write file
				tree = ET.ElementTree(annotation)
				drive, path_and_file = os.path.splitdrive(filename)
				path, file = os.path.split(path_and_file)
				file = file.split(".jpg")[0] + ".xml"
				#print(os.path.join(folderPathAnnotations, file))
				tree.write(os.path.join(folderPathAnnotations, file))
