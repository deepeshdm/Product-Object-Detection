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
