import cv2
import time
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
tf.gfile = tf.io.gfile
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Set page configs.
st.set_page_config(page_title="Product Object Detection", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 50px;font-weight: 350;font-family:Cursive "> Product Detection </p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "&nbsp; Upload your Image below and our ML model will "
    "classify and name each product inside the Image", unsafe_allow_html=True
)

# Example Image
st.image(image="./examples/combine.png")
st.markdown("</br>", unsafe_allow_html=True)

# -------------Body Section------------------------------------------------

# Upload the Image
content_image = st.file_uploader(
    "Upload Content Image (PNG & JPG images only)", type=['png', 'jpg', 'jpeg'])

st.markdown("</br>", unsafe_allow_html=True)
st.warning('NOTE : You need atleast Intel i3 with 8GB memory for proper functioning of this application. ' +
           ' All Images are resized to 420x420')

if content_image is not None:

    with st.spinner("Scanning the Image...will take about 10-15 secs"):

        content_image = Image.open(content_image)

        content_image = np.array(content_image)

        # Resize image to 420x420
        content_image = cv2.resize(content_image, (420, 420))

        # ---------------Detection Phase-------------------------

        # # Path of the pre-trained TF model
        MODEL_DIR = r"./trained-model/saved_model"

        # # Path of the LabelMap file
        PATH_TO_LABELS = r"label_map.pbtxt"

        # Decision Threshold
        MIN_THRESH = float(0.60)

        print('Loading model...', end='')
        start_time = time.time()

        # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
        detect_fn = tf.saved_model.load(MODEL_DIR)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        # LOAD LABEL MAP DATA FOR PLOTTINg
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(content_image)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Detect Objects
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_with_detections = content_image.copy()

        # Detected classes
        detected_classes = detections['detection_classes']
        scores = detections['detection_scores']

        print("Detected Classes : ", detected_classes)
        print("Scores : ", scores)

        # ---------------Drawing Phase-------------------------

        classes = {1: "Beer Opener",
                   2: "Charging Cable",
                   3: "ETUI",
                   4: "Red Case",
                   5: "White Case",
                   6: "USB Stick",
                   7: "Webcam Cover"}

        # Find indexes with scores greater than the MIN_THRESH
        score_indexes = [idx for idx, element in enumerate(scores) if element > MIN_THRESH]
        detected_classes = [detected_classes[idx] for idx in score_indexes]
        # Replace numbers with class names
        detected_classes = [classes.get(i) for i in detected_classes]

        # Count the occurences of each object
        detection_count = dict((i, detected_classes.count(i)) for i in detected_classes)
        print("Detected : ", detection_count)

        # Draw the bounding boxes with probability score
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=MIN_THRESH,
            agnostic_mode=False)

        print('Done')

        if image_with_detections is not None:
            # some baloons
            st.balloons()

        col1, col2 = st.columns(2)
        with col1:
            # Display the output
            st.image(image_with_detections)
        with col2:
            st.markdown("</br>", unsafe_allow_html=True)
            st.markdown(f"<h5> Detected : {detection_count} </h5>", unsafe_allow_html=True)
            st.markdown(
                "<b> Your Image is Ready ! Click below to download it. </b> ", unsafe_allow_html=True)

            # convert to pillow image
            img = Image.fromarray(image_with_detections)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            st.download_button(
                label="Download image",
                data=buffered.getvalue(),
                file_name="output.png",
                mime="image/png")
