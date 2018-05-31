import concurrent.futures
from collections import deque
from time import clock

from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import keras.applications
from keras.preprocessing import image

import numpy as np
import cv2
from camera import MyCamera

class detector_parameters():
    def __init__(self, boundaries, stride, num_predictions_stored):
        self.boundaries = boundaries
        self.stride = stride
        self.num_predictions_stored = num_predictions_stored

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


def getSinglePrediction(model, frame):
    """ 
    returns a 2-tuple (class_label, confidence_score)

    The function will alter the contents of frame, so a copy of the
    frame should be used if integrity of the frame must be preserved.
    """
    # show prediction region
    cv2.imshow('prediction region', frame)

    # convert BGR image to RGB image
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert from 3D tensor to 4D tensor.
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x.astype('float64'))
    prediction = model.predict(x)
    prediction = decode_predictions(prediction, top=1)[0][0][1:]
    return prediction


def getTopPredictions(model, det, frame, max_predicitons = 5):
    """
    runs sliding window detection over the frame, returning the 
    class label and confidence score of the highest scoring label

    returns a 2-tuple (class_label, confidence_score) 
    """

    IMAGE_DIMS = 299

    # create lists containing all potential offsets of the detection window
    window_horizontal_range = range(det.params.boundaries['left'], (det.params.boundaries['right'] - IMAGE_DIMS), det.params.stride['horizontal'])
    window_vertical_range = range(det.params.boundaries['top'], (det.params.boundaries['bottom'] - IMAGE_DIMS), det.params.stride['vertical'])

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
            cv2.rectangle(scanning_frame, (det.params.boundaries['left'],  det.params.boundaries['bottom']), 
                                          (det.params.boundaries['right'], det.params.boundaries['top']), (255, 0, 0), 2)
            cv2.rectangle(scanning_frame, (left_shift,  bottom_shift), 
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


class detector():
    def __init__(self, model, params, sliding_window=False):
        self.future = concurrent.futures.Future()
        # used to compute predictions in separate thread
        self.ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.sliding_window = sliding_window
        self.params = params
        self.timings = deque(maxlen=params.num_predictions_stored)
        self.img_deque = deque(maxlen=params.num_predictions_stored)
        self.model = model
        
    def foundSomething(self):
        return self.future.done() or not self.future.running()

    def detect(self, frame):
        self.t0 = clock()
        if self.sliding_window is True:
            self.future = self.ex.submit(getTopPredictions, self.model, self, frame.copy())
        else:
            # get cropped frame for prediction
            cropped_frame = frame[self.params.boundaries['top' ]:self.params.boundaries['bottom'], 
                                  self.params.boundaries['left']:self.params.boundaries['right']]
            cropped_frame = cv2.resize(cropped_frame.copy(), (299,299))
            # draw box around detection region      
            self.future = self.ex.submit(getSinglePrediction, self.model, cropped_frame)
    
    def getPrediction(self):
        try:
            self.best_pred = self.future.result(timeout=1)
            self.timings.append(clock()-self.t0)
            return True
        except concurrent.futures.TimeoutError:
            return False

    def findAveragePrediction(self):
        if self.sliding_window is True:
            self.img_deque.append(self.best_pred[0])
        else:
            self.img_deque.append(self.best_pred)
        # get the average label, average confidence of that label, 
        # and the proporition of img_deque that contains that label
        self.label_average, self.confidence, self.proportion_images_with_label = getAveragePrediction(self.img_deque)

    def drawDetectionRegion(self, frame):
        # get cropped frame for prediction
        cv2.rectangle(frame, (self.params.boundaries['left' ], self.params.boundaries['bottom']), 
                             (self.params.boundaries['right'], self.params.boundaries['top']), (255, 0, 0), 2)
    
    def getMessages(self):
        if self.sliding_window is True:
            pred = self.best_pred[0]
        else:
            pred = self.best_pred
        
        messages = [
            "Current Prediction: {}     ".format(pred[0]), 
            "Current Confidence: {:.3f} ".format(pred[1]), 
            "Average Prediction: {}     ".format(self.label_average),
            "Average Confidence: {:.3f} ".format(self.confidence), 
            "Average Proportion: {:.3f} ".format(self.proportion_images_with_label), 
            "Time taken for last inference: {:.2f} seconds".format(self.timings[-1]), 
            "Average time taken for inference: {:.2f} seconds per frame".format(sum(self.timings)/len(self.timings)),
            "Box Corners: Lower left: {}, Upper right: {}".format( \
                (self.params.boundaries['left' ], self.params.boundaries['bottom']), 
                (self.params.boundaries['right'], self.params.boundaries['top']))
            ]

        if self.sliding_window is True:
            messages+= [ "Detection Window Vertical, Horizontal Stride: %d, %d" \
                            % (self.params.stride['vertical'], self.params.stride['horizontal']) ]

        return messages
    
    def processControls(self):
        key = cv2.waitKey(1)

        # End loop if the Q key is pressed
        if key == ord('q'):
            return 'q'

        # a,s,d,w shift the boundaries of the detection region
        if key == ord('w') and self.params.boundaries['top'] >= self.params.stride['vertical']:
            self.params.boundaries['top'] -= 20
            self.params.boundaries['bottom'] -= 20
        if key == ord('s') and self.params.boundaries['bottom'] <= 480 - self.params.stride['vertical']:
            self.params.boundaries['top'] += 20
            self.params.boundaries['bottom'] +=20
        if key == ord('a') and self.params.boundaries['left'] >= self.params.stride['horizontal']:
            self.params.boundaries['left'] -= 20
            self.params.boundaries['right'] -= 20
        if key == ord('d') and self.params.boundaries['right'] <= 640 - self.params.stride['horizontal']:
            self.params.boundaries['left'] += 20
            self.params.boundaries['right'] += 20

        # increase the vertical stride of detection window
        if key == ord('r') and self.params.stride['vertical'] > 0:
            self.params.stride['vertical'] += 10
        # decrease the vertical stride of detection window
        if key == ord('f') and self.params.stride['vertical'] < 299:
            self.params.stride['vertical'] -= 10

        # increase the horizontal stride of detection window
        if key == ord('t') and self.params.stride['horizontal'] > 0:
            self.params.stride['horizontal'] += 10
        # decrease the horizontal stride of detection window
        if key == ord('g') and self.params.stride['horizontal'] < 299:
            self.params.stride['horizontal'] -= 10

        return None
