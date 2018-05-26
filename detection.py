from __future__ import print_function
import concurrent.futures
from collections import deque
from operator import itemgetter

from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import keras.applications
from keras.preprocessing import image

import numpy as np
import cv2
from MyCamera import MyCamera

"""-------------------- GLOBAL PARAMETERS --------------------"""
# Used to specify camera if multiple cameras are present
CAMERA_CHOICE = 0

# If SLIDING_WINDOW is True, sliding window detection will be used,
# else single instance detection will be used
SLIDING_WINDOW = False

# Used to compute the average prediction.
NUM_PREDICTIONS_STORED = 5

# Used to specify the bounderies of the box for image detection
TOP_EDGE = 80
BOTTOM_EDGE = 400
LEFT_EDGE = 160
RIGHT_EDGE = 640 - 160

# Specifies how much the single instance detection window shifts
# when a, s, d, or w are pressed
SCREEN_SHIFT_AMOUNT = 15

### Sliding detection window parameters ###

# Used to specify the horizontal distance that the detection window moves
# with each iterations. 
HORIZONTAL_WINDOW_STRIDE = 260
# Use 130 for and even 3 horizontal windows
# HORIZONTAL_WINDOW_STRIDE = 130

# Used to specify the vertical distance that the detection window moves
# with each iterations. 
VERTICAL_WINDOW_STRIDE = 100

# When SLIDING_WINDOW is True, these borders specify the outer boundaries
# that the sliding window must stay inside of
DETECTION_LEFT_BORDER = 40
DETECTION_RIGHT_BORDER = 600
DETECTION_TOP_BORDER = 40
DETECTION_BOTTOM_BORDER = 440

""" -------------------- FUNCTIONS -------------------- """

def getAveragePrediction(recent_image_predictions):
    """
    returns a 3-tuple (class_label, average_confidence_score, proportion_of_list_with_label)

    input: recent_image_predictions: list of last N prediction 2-tuples, (class_label, confidence)
    """

    # stores dictionary { class_label : total_confidence_sum }
    image_label_dict = {}

    # get the sums of prediction probabilities with respect to their label
    for image_prediction in recent_image_predictions:
        if image_prediction[0] in image_label_dict:
            image_label_dict[image_prediction[0]] += image_prediction[1]
        else:
            image_label_dict[image_prediction[0]] = image_prediction[1]

    # find the label with the highest total confidence
    best_label, best_confidence = '', 0
    for label, confidence in image_label_dict.items():
        if best_confidence < confidence:
            best_confidence = confidence
            best_label = label
    
    # count the number of times the best label shows up 
    # in the recent predictions
    count = 0
    for label, conf in recent_image_predictions:
        if label == best_label:
            count += 1

    proportion_images = count / len(recent_image_predictions)
    confidence_average = image_label_dict[best_label] / count

    return best_label, confidence_average, proportion_images


def processControls():
    global TOP_EDGE, BOTTOM_EDGE, LEFT_EDGE, RIGHT_EDGE, SCREEN_SHIFT_AMOUNT
    global HORIZONTAL_WINDOW_STRIDE, VERTICAL_WINDOW_STRIDE, SLIDING_WINDOW

    # End loop if the Q key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        return 'q'

    # a,s,d,w shift the boundaries of the detection region
    if key == ord('w') and TOP_EDGE >= SCREEN_SHIFT_AMOUNT:
        TOP_EDGE -= SCREEN_SHIFT_AMOUNT
        BOTTOM_EDGE -= SCREEN_SHIFT_AMOUNT
    if key == ord('s') and BOTTOM_EDGE <= 480 - SCREEN_SHIFT_AMOUNT:
        TOP_EDGE += SCREEN_SHIFT_AMOUNT
        BOTTOM_EDGE += SCREEN_SHIFT_AMOUNT
    if key == ord('a') and LEFT_EDGE >= SCREEN_SHIFT_AMOUNT:
        LEFT_EDGE -= SCREEN_SHIFT_AMOUNT
        RIGHT_EDGE -= SCREEN_SHIFT_AMOUNT
    if key == ord('d') and RIGHT_EDGE <= 640 - SCREEN_SHIFT_AMOUNT:
        LEFT_EDGE += SCREEN_SHIFT_AMOUNT
        RIGHT_EDGE += SCREEN_SHIFT_AMOUNT

    # increase the vertical stride of detection window
    if key == ord('r') and VERTICAL_WINDOW_STRIDE > 0:
        VERTICAL_WINDOW_STRIDE += 10
    # decrease the vertical stride of detection window
    if key == ord('f') and VERTICAL_WINDOW_STRIDE < 299:
        VERTICAL_WINDOW_STRIDE -= 10

    # increase the horizontal stride of detection window
    if key == ord('t') and HORIZONTAL_WINDOW_STRIDE > 0:
        HORIZONTAL_WINDOW_STRIDE += 10
    # decrease the horizontal stride of detection window
    if key == ord('g') and HORIZONTAL_WINDOW_STRIDE < 299:
        HORIZONTAL_WINDOW_STRIDE -= 10

    return None


def getSinglePrediction(model, frame):
    """ 
    returns a 2-tuple (class_label, confidence_score)

    The function will alter the contents of frame, so a copy of the
    frame should be used if integrity of the frame must be preserved.
    """
    # show prediction region
    cv2.imshow('prediction region', frame)

    # convert from 3D tensor to 4D tensor.
    x = np.expand_dims(frame, axis=0)
    # convert BGR image to RGB image
    x = x[:, :, :, [2, 1, 0]]
    x = preprocess_input(x.astype('float64'))
    prediction = model.predict(x)
    prediction = decode_predictions(prediction, top=1)[0][0][1:]
    return prediction


def getTopPredictions(model, frame, max_predicitons = 5):
    """
    runs sliding window detection over the frame, returning the 
    class label and confidence score of the highest scoring label

    returns a 2-tuple (class_label, confidence_score) 
    """
    global DETECTION_LEFT_BORDER, DETECTION_RIGHT_BORDER, DETECTION_TOP_BORDER, DETECTION_BOTTOM_BORDER
    global HORIZONTAL_WINDOW_STRIDE, VERTICAL_WINDOW_STRIDE, SLIDING_WINDOW

    IMAGE_DIMS = 299

    # create lists containing all potential offsets of the detection window
    window_horizontal_range = range(DETECTION_LEFT_BORDER, (DETECTION_RIGHT_BORDER - IMAGE_DIMS), HORIZONTAL_WINDOW_STRIDE)
    window_vertical_range = range(DETECTION_TOP_BORDER, (DETECTION_BOTTOM_BORDER - IMAGE_DIMS), VERTICAL_WINDOW_STRIDE)

    top_predictions, top_bounding_boxes = [], []

    # iterate over all window offsets
    for vertical_shift in window_vertical_range:
        for horizontal_shift in window_horizontal_range:
            # get cropped region for prediction
            top_shift     = vertical_shift
            bottom_shift  = vertical_shift + IMAGE_DIMS
            left_shift    = horizontal_shift
            right_shift   = horizontal_shift + IMAGE_DIMS
            cropped_frame = frame[top_shift:bottom_shift, left_shift:right_shift]

            # Draw rectangles around current detection region and sliding window boundaries
            scanning_frame = frame.copy()
            cv2.rectangle(scanning_frame, (DETECTION_LEFT_BORDER, DETECTION_BOTTOM_BORDER), 
                          (DETECTION_RIGHT_BORDER, DETECTION_TOP_BORDER), (255, 0, 0), 2)
            cv2.rectangle(scanning_frame, (left_shift, bottom_shift), 
                          (right_shift, top_shift), (0, 255, 0), 2)
            cv2.imshow("Scanning window", scanning_frame)

            # get prediction of current region
            prediction = getSinglePrediction(model, cropped_frame)
            top_predictions.append(prediction)
            top_bounding_boxes.append(((left_shift, bottom_shift), (right_shift, top_shift)))

    top_predictions = top_predictions[:max_predicitons]

    #if SLIDING_WINDOW is False:
        #cv2.destroyWindow("Scanning window")
    
    # Do some magic to sort all predictions and bounding boxes by the confidence of each prediction
    predicitions = sorted(zip(top_predictions, top_bounding_boxes), 
                          key=lambda prediction_and_box: prediction_and_box[0][1], reverse=True)

    top_predictions = predicitions[0][0]
    top_bounding_boxes = predicitions[0][1]
    return top_predictions, top_bounding_boxes


def main():
    global TOP_EDGE, BOTTOM_EDGE, LEFT_EDGE, RIGHT_EDGE, SLIDING_WINDOW
    # initialize the camera
    webcam = MyCamera(CAMERA_CHOICE)

    print("Loading model...")
    model = keras.applications.InceptionResNetV2()
    model._make_predict_function()
    # needed to store prediction in future object
    print("Model loaded")

    # stores previous predictions used to determine 
    # the average prediction over an interval
    img_deque = deque(maxlen=NUM_PREDICTIONS_STORED)

    # used to compute predictions in separate thread
    future = concurrent.futures.Future()
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # define the messages to display on the screen
    messages = ["Current Prediction:", "Current Confidence:", "Average Prediction:", 
                "Average Confidence:", "Average Proportion:"]
    controls = ["Q to quit",
                "A, S, D, W to detection window"]

    if SLIDING_WINDOW is True:
        controls += ["R, F to increase\decreate vertical detection stride",
                     "T, G to increase\decreate horizontal detection stride"]

    cv2.namedWindow('prediction region')
    cv2.namedWindow("Scanning window")

    # start main loop
    while True:
        webcam.tick()
        frame = webcam.getColorFrame()
        
        # get cropped frame for prediction
        if SLIDING_WINDOW is False:
            cropped_frame = frame[TOP_EDGE:BOTTOM_EDGE, LEFT_EDGE:RIGHT_EDGE]
            cropped_frame = cv2.resize(cropped_frame, (299,299))
            # draw box around detection region
            cv2.rectangle(frame, (LEFT_EDGE, BOTTOM_EDGE), (RIGHT_EDGE, TOP_EDGE), (255, 0, 0), 2)                
        
        # display image
        cv2.imshow('image', frame)

        # run if prediction on old frame has been found
        if future.done() or not future.running():
            try:
                preds, bounding_boxes = [], []
                best_pred = None

                # get the prediction
                if SLIDING_WINDOW is True:
                    best_pred, bounding_boxes = future.result(timeout=1)
                else:
                    best_pred = future.result(timeout=1)

                # add prediction to queue for averaging
                img_deque.append(best_pred)
                # get the average label, average confidence of that label, 
                # and the proporition of img_deque that contains that label
                label_average, confidence, proportion_images_with_label = getAveragePrediction(img_deque)
                
                label_current      = "Current Prediction: {}     ".format(best_pred[0])
                confidence_current = "Current Confidence: {:.3f} ".format(best_pred[1])
                label_average      = "Average Prediction: {}     ".format(label_average)
                confidence_average = "Average Confidence: {:.3f} ".format(confidence)
                proportion_average = "Average Proportion: {:.3f} ".format(proportion_images_with_label)
                box_corners        = "Box Corners: {}, {}        ".format((LEFT_EDGE, BOTTOM_EDGE), (RIGHT_EDGE, TOP_EDGE))

                messages = [label_current, confidence_current, label_average,
                            confidence_average, proportion_average, box_corners]

                if SLIDING_WINDOW is True:
                    detection_stride = "Detection Window Vertical, Horizontal Stride: %d, %d" \
                                        % (VERTICAL_WINDOW_STRIDE, HORIZONTAL_WINDOW_STRIDE)
                    messages.append(detection_stride)
                else:
                    messages.append("Sliding window is off")

            except concurrent.futures.TimeoutError:
                pass

            print("---------------")
            for message in messages:
                print(message)

            # get the prediction of the current frame in a separate thread
            if SLIDING_WINDOW is True:
                future = ex.submit(getTopPredictions, model, frame.copy())
            else:
                future = ex.submit(getSinglePrediction, model, cropped_frame.copy())

        # draw controls on separate frame
        control_frame = np.zeros((300, 800, 3))
        for message in messages + controls:
            webcam.display_text(control_frame, message)
        cv2.imshow('controls', control_frame)

        # process key press and exit if key pressed is Q
        if processControls() == 'q':
            break

    # clean up windows when done
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
