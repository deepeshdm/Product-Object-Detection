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


The "main.py" python script is where we define the essentials like trained-model path, labelmap path ,Input image path etc. You need to pass the required values as per your system's file system. All the files are included in this repository.

```python
import cv2
from API import make_detection

# Path to Input image. NOTE : Minimum image size is 420x420
IMAGE_PATH = "C:/Users/Deepesh/Desktop/fiverr/training_demo/images/train/test_6.jpeg"

# Path to LabelMap file
LABELS_PATH = "C:/Users/Deepesh/Desktop/Product Detection/label_map.pbtxt"

# Path to pre-trained model 
MODEL_PATH = "C:/Users/Deepesh/Desktop/Product Detection/trained-model/saved_model"

# -----------------------------------------------------------------

OUTPUT_IMG = make_detection(IMAGE_PATH, MODEL_PATH, LABELS_PATH)

# SAVE OUTPUT IMAGE
print("Saving the output image locally...")
Save_Path = "product_detection.jpg"
cv2.imwrite(Save_Path, OUTPUT_IMG)

# DISPLAY OUTPUT
cv2.imshow("RESULT", OUTPUT_IMG)
cv2.waitKey(0)
```

## Web Interface

The Web Interface is made using streamlit, you can run it locally by executing the following commands :

**🔥 Official Website :** https://share.streamlit.io/deepeshdm/product-object-detection/User_Interface.py

<div align="center">
  <img src="/examples/ui.png"  width="90%"/>
</div>
<br/>


1. Git clone the repository on your system. This will download the pre-trained model and required files on your computer.
```
git clone https://github.com/deepeshdm/Product-Object-Detection.git
```

2. Install the required dependencies to run the app
```
cd Product-Object-Detection

pip install -r requirements.txt
```

3. Start the streamlit server on specified port

```
streamlit run User_Interface.py --server.port 4000
```



## Important

If you encounter an error - 'tensorflow has no attribute gfile' , then you'll have to make changes to your site-packages present locally.
- Follow this : https://github.com/tensorflow/tensorflow/issues/31315
- A quick fix for this is to put the following line after you import tensorflow:
```python
tf.gfile = tf.io.gfile
```



