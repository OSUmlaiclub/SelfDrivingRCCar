import cv2
from collections import deque

def showFrames(frame_names, frames):
    for i in range(len(frames)):
        cv2.imshow(frame_names[i], frames[i])

# Webcam object used to interact with frames
class MyCamera:
    # Initializes camera and Mats
    def __init__(self, camera_number=0, display_trackbars=False, num_old_frames_stored=4):
        self.cap = cv2.VideoCapture(camera_number)
        self.text_display_offset = 20

        self.old_frames = deque(maxlen=num_old_frames_stored)

        ret, self.frame1 = self.cap.read()
        while ret is False:
            ret, self.frame1 = self.cap.read()

        for i in range(num_old_frames_stored):
            self.old_frames.append(self.frame1.copy())

        # frame conversion properties
        self.min_canny_threshold = 30
        self.max_canny_threshold = 150
        self.min_difference_threshold = 40
        self.max_difference_threshold = 220

        self.trackbar_window_name = None
        if display_trackbars:
            self.initialize_trackbars()

    def set_res(self, x, y):
        self.cap.set(3, int(x))
        self.cap.set(4, int(y))
        return str(self.cap.get(3)),str(self.cap.get(4))

    def getColorFrame(self):
        return self.frame1.copy()
    
    def getGreyFrame(self, blur_kernel=(3, 3)):
        self.grey = cv2.cvtColor(self.frame1.copy(), cv2.COLOR_BGR2GRAY)
        grey_blurred = cv2.blur(self.grey, blur_kernel)
        return grey_blurred

    def getCannyFrame(self, dilation_kernel=(3, 3), dilation_iterations=1):
        canny = cv2.Canny(self.getGreyFrame(), self.min_canny_threshold, self.max_canny_threshold)
        for i in range (dilation_iterations):
            canny = cv2.dilate(canny, dilation_kernel)
        return canny

    def getThresholdFrame(self, dilation_kernel=(3, 3), dilation_iterations=1, 
                          min_thresh=None, max_thresh=None):
        t_gray1 = cv2.cvtColor(self.frame1.copy(), cv2.COLOR_BGR2GRAY)
        t_gray2 = cv2.cvtColor(self.old_frames[-1].copy(), cv2.COLOR_BGR2GRAY)
        cv2.blur(t_gray1, (3, 3))
        cv2.blur(t_gray2, (3, 3))
        difference = cv2.absdiff(t_gray1, t_gray2)
        ret, threshold = cv2.threshold(difference, self.min_difference_threshold, self.max_difference_threshold, cv2.THRESH_BINARY)
        for i in range(dilation_iterations):
            threshold = cv2.dilate(threshold, dilation_kernel)
        return threshold

    def initialize_trackbars(self):
        # Needed to initialize trackbars
        def nothing(x):
            x = x
            return x 

        # create trackbars for thresholds
        self.trackbar_window_name = 'trackbars'
        cv2.namedWindow(self.trackbar_window_name)
        cv2.createTrackbar('min_canny_threshold', self.trackbar_window_name, self.min_canny_threshold, 255, nothing)
        cv2.createTrackbar('max_canny_threshold', self.trackbar_window_name, self.max_canny_threshold, 255, nothing)
        cv2.createTrackbar('min_difference_threshold', self.trackbar_window_name, self.min_difference_threshold, 255, nothing)
        cv2.createTrackbar('max_difference_threshold', self.trackbar_window_name, self.max_difference_threshold, 255, nothing)

    # get current positions of trackbars
    def process_trackbars(self):
        self.min_canny_threshold = cv2.getTrackbarPos('min_canny_threshold', self.trackbar_window_name)
        self.max_canny_threshold = cv2.getTrackbarPos('max_canny_threshold', self.trackbar_window_name)
        self.min_difference_threshold = cv2.getTrackbarPos('min_difference_threshold', self.trackbar_window_name)
        self.max_difference_threshold = cv2.getTrackbarPos('max_difference_threshold', self.trackbar_window_name)

    def display_text(self, image, text, color=(255, 255, 255)):
        cv2.putText(image, text, (5, self.text_display_offset), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 7)
        cv2.putText(image, text, (5, self.text_display_offset), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)
        self.text_display_offset += 25
        return image
        
    # update frames
    def tick(self):
        self.old_frames.append(self.frame1)
        ret, self.frame1 = self.cap.read()
        while ret is False:
            ret, self.frame1 = self.cap.read()
        self.text_display_offset = 20

        if self.trackbar_window_name is not None:
            self.process_trackbars()


def main():
    print("Initializing camera object")
    camera = MyCamera(0, display_trackbars=True)
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        camera.tick()
        color_frame = camera.getColorFrame()
        threshold = camera.getThresholdFrame()
        canny = camera.getCannyFrame()

        frame_names = ["color", "threshold", "canny"]
        frames = [color_frame, threshold, canny]
        showFrames(frame_names, frames)

if __name__ == "__main__":
    main()
