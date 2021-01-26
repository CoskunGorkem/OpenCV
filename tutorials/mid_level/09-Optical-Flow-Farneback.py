import cv2 as cv
import numpy as np

import argparse




def main(video):

    # The video feed is read in as
    # a VideoCapture object
    cap = cv.VideoCapture(video)
    
    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    ret, first_frame = cap.read()
    first_frame = cv.resize(first_frame, (360, 200))
    
    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    while (cap.isOpened()):
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        ret, frame = cap.read()
        
        #Resizing the image to speed up the process
        #NOTE ---> This will decrease the visual accuracy.
        frame = cv.resize(frame, (360, 200))
        
        # Opens a new window and displays the input
        # frame
        cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)


        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame


        # Updates previous frame
        prev_gray = gray
        cv.imshow("input", cv.resize(frame, (720, 400)))
        cv.imshow("dense optical flow", cv.resize(rgb, (720, 400)))
        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
  
    args = parser.parse_args()
    main(args.video)

