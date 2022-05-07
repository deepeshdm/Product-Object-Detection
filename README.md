## Product Object Detection

A Object detection system for localizing and classifying 7 different classes of objects. The model used is the 'SSD-MOBILENET-V1',which has processing speed of 48ms and a Mean Average Precision (mAP) of 29.

<div float="left" align="center">
<img src="/examples/result1.jpeg"  width="30%"/>
<img src="/examples/result2.jpeg"  width="30%"/>
<img src="/examples/result.jpeg"  width="30%"/>
</div>


## To Run (Locally)

1. Git clone the repository on your system. This will download the pre-trained model and required files on your computer.
```
git clone https://github.com/deepeshdm/Product-Object-Detection.git
```

2. Install the required dependencies to run the app
```
cd Product-Object-Detection

pip install -r requirements.txt
```

3. Open the "main.py" file , pass the required values to the function , Execute the file.

```
python main.py
```

## Usage Description


The "API.py" python script contains 3 main functions responsible for object detections :

1. detect_pastries( )

This function is straight-forward, it takes the path of an Image with other parameters , detects the pastries present in the image and draws bounding boxes around them,after that it simply displays the output Image. It does'nt return anything else.

```python
# Path to Input Image
IMAGE_PATH = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/train/IMG_0008.JPG'

# Path to pre-trained model
SAVED_MODEL_PATH = r'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/saved_model'

# Path to 'label_map.pbtxt' file
PATH_TO_LABELS = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/label_map.pbtxt'

# Object detection threshold
THRESHOLD = 0.5

# Path to save the output Image with name and extension
SAVE_PATH = r"C:/Users/dipesh/Desktop/fiverr-hasijayawardana/outputs/output.jpeg"

from API import detect_pastries
detect_pastries(IMAGE_PATH, SAVED_MODEL_PATH, PATH_TO_LABELS,THRESHOLD,SAVE_PATH)
```
   
2. detect_objects( )

This function takes the path of an Image with other parameters , detects the pastries present in the image and draws bounding boxes around them,after that it does'nt display the Image like the above function. It returns 2 things,first the output image as numpy array and second it returns a dictionary of Top 10 objects detected by the model with their scores,this dictionary object can be used during production for some other task where we want to process the output,like the billing system.


```python
# Path to Input Image
IMAGE_PATH = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/train/IMG_0008.JPG'

# Path to pre-trained model
SAVED_MODEL_PATH = r'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/saved_model'

# Path to 'label_map.pbtxt' file
PATH_TO_LABELS = 'C:/Users/dipesh/Desktop/fiverr-hasijayawardana/Pastry Detection/label_map.pbtxt'

# Object detection threshold
THRESHOLD = 0.5

# Path to save the output Image with name and extension
SAVE_PATH = r"C:/Users/dipesh/Desktop/fiverr-hasijayawardana/outputs/output.jpeg"

import cv2
from API import detect_objects

# Numpy array of output Image and Dictionary of Top 10 detected objects with scores
Output_Image, Detected_Objects = detect_objects(IMAGE_PATH, SAVED_MODEL_PATH, PATH_TO_LABELS, THRESHOLD)

print(Detected_Objects)
cv2.imshow("OUTPUT", Output_Image)
cv2.waitKey()
```

3. detect_webcam( ) 

This function is for the webcam object detection. When you execute this function,It'll automatically activate the webcam and start detecting objects frame by frame. The current speed is 3FPS,but can be further increased by running on GPU/TPU and reducing preprocessing/postprocessing of the Images.

```python
from API import detect_webcam
import tensorflow as tf

# Path to pre-trained model
SAVED_MODEL_PATH = \
    r'C:\Users\dipesh\fiverr-hasijayawardana-object-detector\saved_model'

# Path to 'label_map.pbtxt' file
PATH_TO_LABELS = \
    r'C:\Users\dipesh\fiverr-hasijayawardana-object-detector\label_map.pbtxt'

# Object detection threshold
THRESHOLD = 0.5

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
print("Loading model...this will take a minute")
detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)

detect_webcam(detect_fn, PATH_TO_LABELS, THRESHOLD)
```


### NOTE : When you execute the python files, keep an eye on the terminal , it'll print the logs while the model is detecting objects.

## Important

If you encounter an error - 'tensorflow has no attribute gfile' , then you'll have to make changes to your site-packages present locally.
- Follow this : https://github.com/tensorflow/tensorflow/issues/31315
- A quick fix for this is to put the following line after you import tensorflow:
```python
tf.gfile = tf.io.gfile
```



