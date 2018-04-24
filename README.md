<h1> Impy (Images in python) </h1>
<p>Impy is a library used for deep learning projects that make use of image datasets.</p>
<ul>
  <li><strong>Email: </strong>lozuwaucb@gmail.com</li>
  <li><strong>Bug reports: </strong>https://github.com/lozuwa/impy/issues</li>
</ul>
<p>It provides:</p>
<ul>
  <li>Data augmentation methods for images with bounding boxes (the bounding boxes are also affected by the transformation so you don't have to label again.)</li>
  <li>Fast image preprocessing methods useful in a deep learning context. E.g: if your image is too big you need to divide it into patches.</li>
</ul>

<p>For now the installation requires you to clone the repository. Follow the next steps:</p>

<ol>
  <li>Go to your package path. E.g: for anaconda go to ../Anaconda3/envs/YOUR_ENVIRONMENT/lib/pythonVERSION_OF_YOUR_PYTHON/site-packages/</li>
  <li>Clone the library with the following command git clone  https://github.com/lozuwa/impy.git</li>
  <li>Test the library by working on one of the tutorials.</li>
</ol>

<h1> Tutorial </h1>
<p>Impy has multiple features that allow you to solve several different problems with a few lines of code. To make things clear we are going to tackle problems you might find 
while working on Deep learning for computer vision. </p>
<p>We are going to work with the following mini-dataset of cars and pedestrians. The images have xml annotations that contain the coordinates of the bounding boxes that enclose the cars and 
pedestrians in the images.</p>

<img src="tests/cars_dataset/images/cars0.png" alt="cars_dataset" height="42" width="42"></img>

<h2>Object localization</h2>
<p>In this section we are going to solve problems related with object localization.</p>
<h3>Images are too big</h3>
<p>One common problem in Computer Vision and CNNs is dealing with big images. Let's sample one of the images from our mini-dataset: </p>

<img src="tests/cars_dataset/images/cars1.png" alt="cars_dataset" height="42" width="42"></img>

<p>This image's size is 3840x2160. It is too big for training, it will lower the size of your mini-batch hyperparameter or simply your computer will not have enough memory for it.</p>
<p>In order to solve this problem and make training feasable, we are going to crop ROIs of the image to decrease its size. In my case, images of 1032x1032 pixels are small enough for training. But how do we do this, it involves a lot of algorithms. We are going to use Impy.</p>

```python
from impy.ImageLocalizationDataset import *

def main():
 # Define the path to images and annotations
 images_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "images")
 annotations_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "annotations", "xmls")
 # Define the name of the dataset
 dbName = "CarsDataset"
 # Create an object of ImageLocalizationDataset
 imda = ImageLocalizationDataset(images = images_path, 
                                 annotations = annotations_path,
                                 databaseName = dbName)
 # Reduce the dataset by Rois of smaller size with the shape 1032x1032
 images_output_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "images_reduced")
 annotations_output_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "annotations_reduced", "xmls")
 imda.reduceDatasetByRois(offset = 1032,
                          outputImageDirectory = images_output_path,
                          outputAnnotationDirectory = annotations_output_path)

if __mame__ == "__main__":
 main()
```

<p>The previous script will create a new set of images and annotations with the size specified by offset and will include the maximum number of annotations possible so you will end up with an optimal number of
data points. Let's see the results of one of the images: </p>

<img src="tests/cars_dataset/images/cars0.png" alt="cars_dataset" height="42" width="42"></img>

<p>As you can see the annotations have been maintained and small crops of the big image are now available. Our problem is solved.</p>

<h3>Data augmentation for bounding boxes</h3>
<p>Another common problem in Computer Vision and CNNs for object localization is data augmentation. Specifically space augmentations (e.g: scaling, cropping, rotation, etc.). For this you would usually make
a custom script. But with impy we can make it easier.</p>

<p>First, let's create a configuration file. You can use one of the templates available in the confs folder.</p>

```json
{
	"multiple_image_augmentations": {
		"Sequential": [
			{
				"image_color_augmenters": {
					"Sequential": [
						{
							"sharpening": {
								"weight": 0.2,
								"save": true
							}
						},
					]
				}
			},
			{
				"bounding_box_augmenters": {
					"Sequential": [
						{
							"scale": {
								"size": [500, 500],
								"interpolationMethod": 1,
								"save": false
							}
						},
						{
							"verticalFlip": {
								"save": true
							}
						}
					]
				}
			},
			{
				"image_color_augmenters": {
					"Sequential": [
						{
							"histogramEqualization":{
								"equalizationType": 1,
								"save": true
							}
						}
					]
				}
			},
			{
				"bounding_box_augmenters": {
					"Sequential": [
						{
							"horizontalFlip": {
								"save": true
							}
						},
						{
							"crop": {
								"size": [0.7, 0.9],
								"save": true
							}
						}
					]
				}
			},
		]
	}
}
```

<p>Let's analize the configuration file step by step. Currently, this is the most complex type of data augmentation you can achieve with the library.</p>
<p>Note the file starts with "multiple_image_augmentations", then a "Sequential" key follows. Inside "Sequential" we define an array.</p>
<p>This is important, each element of the array is a type of augmenter.</p>
<p>The first augmenter we are going to define is a "image_color_agumenters" which is going to execute a sequence of color augmentations. In this case, we have defined only one type of color augmentation which
is sharpening with a weight of 0.2 and we choose to save it.</p>
<p>After the color augmentation, we have defined a "bounding_box_augmenters" which is going to execute a "scale" augmentation which we choose not to save followed by a "verticalFlip" which we do choose to save.</p>
<p>So, we want to keep going. So we define two more types of image augmenters. Another "image_color_augmenters" which applies "histogramEqualization" to the image.</p>
<p>Finally, we define a "bounding_box_agumeneters" that applies a "horizontalFlip" and a "crop" augmentation.</p>
<p>As you have seen we can define any type of crazy configuration and augment our images with the available methods. Get creative and define your own data augmentation pipelines.</p>

<p>Once the configuration file has been created, we can apply the data augmentation pipeline with the following code.</p>

```python
from impy.ImageLocalizationDataset import *

def main():
 # Define the path to images and annotations
 images_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "images")
 annotations_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "annotations", "xmls")
 # Define the name of the dataset
 dbName = "CarsDataset"
 # Create an object of ImageLocalizationDataset
 imda = ImageLocalizationDataset(images = images_path, 
                                 annotations = annotations_path,
                                 databaseName = dbName)
 # Apply data augmentation by using the following method of the ImageLocalizationDataset class.
 configuration_file = os.path.join(os.getcwd(), "tests", "cars_dataset", "augmentation_configuration.json")
 images_output_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "images_augmented")
 annotations_output_path = os.path.join(os.getcwd(), "tests", "cars_dataset", "annotations_augmented", "xmls")
 imda.applyDataAugmentation(configurationFile = configuration_file,
                          outputImageDirectory = images_output_path,
                          outputAnnotationDirectory = annotations_output_path)

if __mame__ == "__main__":
 main()
```

<p>These are the results:</p>

<img src="tests/cars_dataset/images/cars0.png" alt="cars_dataset" height="42" width="42"></img>

