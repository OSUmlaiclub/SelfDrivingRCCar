import keras.applications
import numpy as np
import cv2

from camera import MyCamera
from detector import detector, detector_parameters

"""-------------------- GLOBAL PARAMETERS --------------------"""
# Used to specify camera if multiple cameras are present
CAMERA_CHOICE = 0

# If SLIDING_WINDOW is True, sliding window detection will be used,
# else single instance detection will be used
SLIDING_WINDOW = False

# Used to compute the average prediction.
NUM_PREDICTIONS_STORED = 5

# Used to specify the bounderies of the box for single instance image detection
TOP_BOUNDARY = 80
BOTTOM_BOUNDARY = 400
LEFT_BOUNDARY = 160
RIGHT_BOUNDARY = 640 - 160

### Sliding detection window parameters ###

# Used to specify the horizontal distance that the detection window moves
# with each iterations. 
HORIZONTAL_STRIDE = 260
# Use 130 for and even 3 horizontal windows
# HORIZONTAL_STRIDE = 130

# Used to specify the vertical distance that the detection window moves
# with each iterations. 
VERTICAL_STRIDE = 100

# When SLIDING_WINDOW is True, these borders specify the outer boundaries
# that the sliding window must stay inside of
SLIDING_LEFT_BOUNDARY = 40
SLIDING_RIGHT_BOUNDARY = 600
SLIDING_TOP_BOUNDARY = 40
SLIDING_BOTTOM_BOUNDARY = 440

def main():
    # initialize the camera
    webcam = MyCamera(CAMERA_CHOICE)

    # initialize the model
    print("Loading model...")
    model = keras.applications.InceptionResNetV2()
    # needed to store prediction in future object
    model._make_predict_function()
    print("Model loaded")

    # specify detection parameters
    if SLIDING_WINDOW is False:
        window_boundaries = {'left':LEFT_BOUNDARY, 'right':RIGHT_BOUNDARY, 
                             'top':TOP_BOUNDARY,   'bottom':BOTTOM_BOUNDARY }      
        window_stride = {'horizontal':10, 'vertical':10}
    else:
        window_boundaries={'left':SLIDING_LEFT_BOUNDARY, 'right':SLIDING_RIGHT_BOUNDARY, 
                           'top':SLIDING_TOP_BOUNDARY,   'bottom':SLIDING_BOTTOM_BOUNDARY }
        window_stride =  {'horizontal':HORIZONTAL_STRIDE, 'vertical':VERTICAL_STRIDE}

    params = detector_parameters(boundaries=window_boundaries, 
                                 stride=window_stride,
                                 num_predictions_stored=NUM_PREDICTIONS_STORED)

    det = detector(model, params=params, sliding_window=SLIDING_WINDOW)

    # define the messages to display on the screen
    messages = ["Current Prediction:", "Current Confidence:", "Average Prediction:", 
                "Average Confidence:", "Average Proportion:"]
    controls = ["Q to quit", 
                "A, S, D, W to move detection window"]

    cv2.namedWindow('prediction region')

    if SLIDING_WINDOW is True:
        cv2.namedWindow("Scanning window")
        controls += ["R, F to increase\decreate vertical detection stride",
                     "T, G to increase\decreate horizontal detection stride"]

    # start main loop
    while True:
        # capture and get next frame
        webcam.tick()
        frame = webcam.getColorFrame()               
        
        # run if prediction on old frame has been found
        if det.foundSomething():
            if det.getPrediction():
                det.findAveragePrediction()
                messages = det.getMessages()
            
            print("-----------------") 
            print(*messages, sep='\n')

            det.detect(frame)

        det.drawDetectionRegion(frame)

        # display image
        cv2.imshow('image', frame)

        # draw controls on separate frame
        control_frame = np.zeros((500, 800, 3))
        webcam.display_texts(control_frame, messages + controls)
        cv2.imshow('controls', control_frame)

        # process key press and exit if key pressed is Q
        if det.processControls() == 'q':
            break

    # clean up windows when done
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
