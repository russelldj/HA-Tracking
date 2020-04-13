#! /usr/bin/python3
# From Python
# It requires OpenCV installed for Python

# this should be run from the openpose <root>/examples/
import sys
import cv2
import os
from sys import platform
import argparse
import time
import pdb
import numpy as np
import pdb
import copy

# import the mask module
TTM_ROOT = "/home/drussel1/dev/TTM/TTM"
sys.path.append(TTM_ROOT)
from tools import tools as TTM_tools
from tools import mask as TTM_mask


INJECTION_PREPARATION_FNAME = "{}/data/TTM-data/processed/EIMP/new-EIMP-mask-RCNN-detections/Injection_Preparation.mp4.h5".format(TTM_ROOT)
IMAGE_SHAPE = (640, 980)
IDS = [0, 1]
BOUNDARY_PX = 50 # the additional context around the hand to add in pixels
X = 0
Y = 1
W = 2
H = 3
VISUALIZE = False
BIG_NUMBER = 1000000

class keypointExtractor():
    def __init__(self):
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
                self.op = op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
                self.op = op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
        
        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()
        
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"
        params["hand"] = True
        params["hand_detector"] = 2
        # try to figure out why this is having issues
        #params["body_disable"] = True
        
        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item
        
        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()
        
        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def computeKeypoints(self, input_video=None, input_h5=None, output_dir=None):
        # Up till now this is just openpose initialization
        vidCapture = cv2.VideoCapture(input_video)
        if not vidCapture.isOpened():
            raise ValueError("The capture failed to open from filename {}".format(VIDEO_INPUT_FNAME))
        for frame_index in range(BIG_NUMBER): # loop for an arbitrarily large number of times
            ret, imageToProcess = vidCapture.read()
            if not ret:
                break
            #TODO pick a naming convention
            hand_rectangles, imageToProcess = self.createBoxes(input_h5, imageToProcess, frame_index)

            # Create new datum
            datum = self.op.Datum()
            datum.cvInputData = imageToProcess
            datum.handRectangles = hand_rectangles
            
            # Process and display image
            self.opWrapper.emplaceAndPop([datum])
            print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
            print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
            VISUALIZE = False
            if VISUALIZE:
                cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
                cv2.waitKey(10)
            else:
                if frame_index % 100 == 0:
                    cv2.imwrite("{}/vis_{:09d}.jpeg".format(output_dir, frame_index), datum.cvOutputData)
                np.save("{}/keypoints_{:09}.npy".format(output_dir, frame_index), datum.handKeypoints)
   

    def loadMask(self, filename=INJECTION_PREPARATION_FNAME, index=0):
        """
        find the location of the hand from Mask R-CNN output
        """
    
        # this is a list which is the number of masks long
        mask_contours = TTM_mask.extract_masks_one_frame(filename, index)
        
        hand_squares = [] 
        for mask_contour in mask_contours:
            output = self.extractBoundingRectangle(mask_contour)
            output = self.rectToSquareRect(output)
            hand_squares.append(output)

        # make some sort of ordering

        return hand_squares


    def extractBoundingRectangle(self, contour):
        """
        countour is a single opencv contour item
        """
        mask_array = TTM_mask.contour_to_biggest_mask(IMAGE_SHAPE, [contour])# this is supposed to be a list of contours
        y_inds, x_inds = np.nonzero(mask_array)
        if len(x_inds) == 0:
            return [IMAGE_SHAPE[0], IMAGE_SHAPE[1], 0, 0] #This is the null box, zero sized in the lower left corner
        
        x = min(x_inds) - BOUNDARY_PX
        y = min(y_inds) - BOUNDARY_PX
        w = max(x_inds) - min(x_inds) + 2 * BOUNDARY_PX
        h = max(y_inds) - min(y_inds) + 2 * BOUNDARY_PX
    
        return [x, y, w, h]


    def createBoxes(self, h5_file, imageToProcess=np.zeros((480, 640, 3)), frame_index=0):
        """
        This by some means needs to load the hands and automatically select a rectangle
        """
        #VIDEO_INPUT_FNAME = "/home/russeldj/dev/TTM-data/raw/EIMP/Bag_Mask_Ventilation-Wv78jVhSFTI.mp4" 
        USE_MASK_RCNN = True
    
        if USE_MASK_RCNN:
            rects = self.loadMask(h5_file, frame_index)
        else:
            rects = [list(cv2.selectROI(imageToProcess))] # this is just massaging it into the form of the other function
        
        # visualize for sanity checking
        if VISUALIZE:
            display = np.copy(imageToProcess)
            black = (0, 0, 0)
            width = 5
            for rect in rects: 
                tl = (int(rect[X]), int(rect[Y]))
                br = (int(rect[X] + rect[W]), int(rect[Y] + rect[H]))
                cv2.rectangle(display, tl, br, black, width)
            cv2.imshow("", display)
            cv2.waitKey(0)
            
            cv2.destroyAllWindows()
        #TODO this should really return it in the form that op needs
        #TODO sort by the left side for now, but improve later
        rects = sorted(rects, key=lambda rect : rect[0]) # sort by the x location
        #HACK assume that there will only be two rectangles

        # format: self.op.Rectangle(*rect), # expand to fill the arguments
        #TODO swap this back to the other way
        if len(rects) >= 2:
            left_rect = rects[0]
            right_rect = rects[1]
            # currently this takes the same box for left and right but this will be made more inteligent
            #this appears to be x1, y1, w, h with w needing to be equal to h
        elif len(rects) == 1:
            left_rect = rects[0]
            right_rect = rects[0]
        else:
            left_rect = [0, 0, 0, 0]
            right_rect = [0, 0, 0, 0]



        hand_rectangles = [[self.op.Rectangle(*left_rect), self.op.Rectangle(*right_rect)]]
        
        return hand_rectangles, imageToProcess
   

    def rectToSquareRect(self, rect):
        """
        takes an [x, y, w, h] rect and returns one in the same format which is a square with the same center
        """
        output_rect = copy.copy(rect)
        
        if output_rect[W] > output_rect[H]:
            # subtract half the difference between w and h to keep it centered
            output_rect[Y] -= (output_rect[W] - output_rect[H]) / 2.0
        else: 
            output_rect[X] -= (output_rect[H] - output_rect[W]) / 2.0
        
        max_size = max(output_rect[W], output_rect[H])
        output_rect[W] = max_size
        output_rect[H] = max_size
        return output_rect

# run the extraction
if __name__ == "__main__":
    keypoint_extractor = keypointExtractor()
    for i in range(11,21): 
        video_name = '/home/drussel1/data/ADL/ADL_videos/P_{:02d}.MP4'.format(i)
        h5_name = '/home/drussel1/data/ADL/detections/5_29_19/P_{:02d}.h5'.format(i) 
        output_dir = '/home/drussel1/data/ADL/openpose_keypoints/keypoint_{:02d}'.format(i)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        assert(os.path.isfile(video_name))
        assert(os.path.isfile(h5_name))
        keypoint_extractor.computeKeypoints(video_name, h5_name, output_dir)
