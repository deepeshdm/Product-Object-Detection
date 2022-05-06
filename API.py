
import tensorflow as tf
tf.gfile = tf.io.gfile
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import cv2


def make_detection(IMG_PATH, MODEL_PATH, LABELS_PATH):
    """
      Args:
          IMG_PATH (str): Path of the receipt image.
          MODEL_PATH (str): Path of trained model.
          SAVE_PATH (str): Path where to save the output image. Default is current directory.
          LABELS_PATH (str): Path to label_map.pbtxt file.
      Returns:
          Output image : The output image as numpy array
    """

    # ------------------------------------------------------------------------

    # LOAD THE MODEL
    print('Loading model...', end='')
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    detect_fn = tf.saved_model.load(MODEL_PATH)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(round(elapsed_time, 3)))

    # LOAD LABEL MAP DATA FOR PLOTTING
    category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH,
                                                                        use_display_name=True)

    # ------------------------------------------------------------------------

    print('Running inference for {}... '.format(IMG_PATH), end='')

    image = cv2.imread(IMG_PATH)

    # Resize image to 420x420
    print("Resizing image to 420x420")
    image = cv2.resize(image, (420,420))

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    image_with_detections = image.copy()

    print(detections['detection_classes'])
    print(detections['detection_scores'])

    # ------------------------------------------------------------------------

    # DECISION THRESHOLD
    Threshold = 0.65

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=Threshold,
        agnostic_mode=False)

    print('Done')
    return image_with_detections