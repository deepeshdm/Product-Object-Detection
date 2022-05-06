
import cv2
from API import make_detection

classes = ["Beer Opener",
            "Charging Cable",
            "ETUI",
            "Red Case",
            "USB Stick",
            "Webcam Cover",
            "White Case","test"]

# Path to Input image. NOTE : Minimum image size is 420x420
IMAGE_PATH = "C:/Users/Deepesh/Desktop/fiverr/training_demo/images/train/test_6.jpeg"

# PROVIDE PATH TO LABEL MAP
LABELS_PATH = "C:/Users/Deepesh/Desktop/Product Detection/label_map.pbtxt"

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
