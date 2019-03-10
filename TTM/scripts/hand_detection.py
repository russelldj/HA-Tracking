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

# import the mask module
sys.path.append("../../../../..")
from tools import tools as TTM_tools
from tools import mask as TTM_mask

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

    def computeKeypoints(self):
        
        
        # Up till now this is just openpose initialization
        VIDEO_INPUT_FNAME = "../../../../../data/TTM-data/raw/EIMP/Injection_Preparation.mp4" 
        vidCapture = cv2.VideoCapture(VIDEO_INPUT_FNAME)
        _, imageToProcess = vidCapture.read()
        if not vidCapture.isOpened():
            raise ValueError("The capture failed to open from filename {}".format(VIDEO_INPUT_FNAME))
        
        # select the rectangles, which are in [x, y, w, h] parametarization
        # the only modification that needs
        rect, imageToProcess = self.createBox(imageToProcess)
    
        # currently this takes the same box for left and right but this will be made more inteligent
        #this appears to be x1, y1, w, h with w needing to be equal to h
        handRectangles = [
            [
            #op.Rectangle(200.0, 70.0, 500.0, 500.0),
            self.op.Rectangle(*rect), # expand to fill the arguments
            #op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
            self.op.Rectangle(*rect) # expand to fill the arguments
            ]#,
        ]
        
        # Create new datum
        datum = self.op.Datum()
        datum.cvInputData = imageToProcess
        datum.handRectangles = handRectangles
        
        # Process and display image
        self.opWrapper.emplaceAndPop([datum])
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(0)
   

    def loadMask(self, filename=""):
        """
        find the location of the hand from Mask R-CNN output
        """
        IMAGE_SHAPE = (640, 980)
        INJECTION_PREPARATION_FNAME = "../../../../../data/TTM-data/processed/EIMP/new-EIMP-mask-RCNN-detections/Injection_Preparation.mp4.h5"
        IDS = [0, 1]
        BOUNDARY = 50 # the additional context around the hand to add in pixels
    
        # this is a list which is the number of masks long
        mask_contours = TTM_mask.extract_masks_one_frame(INJECTION_PREPARATION_FNAME, 0)
        
        #HACK
        # Only take the second hand 
        for mask_contour in mask_contours[1:2]:
            mask_array = TTM_mask.contour_to_biggest_mask(IMAGE_SHAPE, [mask_contour])# this is supposed to be a list of contours
            #np.save("maskarray.npy", mask_array)
            y_inds, x_inds = np.nonzero(mask_array)
        
        x = min(x_inds) - BOUNDARY
        y = min(y_inds) - BOUNDARY
        w = max(x_inds) - min(x_inds) + 2 * BOUNDARY
        h = max(y_inds) - min(y_inds) + 2 * BOUNDARY
    
        return [x, y, w, h]
    
        #visualization = TTM_mask.draw_mask(np.zeros(IMAGE_SHAPE, dtype=np.uint8), mask_contours, IDS)
        #cv2.imshow("", visualization)
        #cv2.waitKey(0)
        #mask_array = TTM_mask.contour_to_biggest_mask(IMAGE_SHAPE, mask_list)
    
    
    def createBox(self, imageToProcess=np.zeros((480, 640, 3))):
        """
        This by some means needs to load the hands and automatically select a rectangle
        """
        VIDEO_INPUT_FNAME = "/home/russeldj/dev/TTM-data/raw/EIMP/Bag_Mask_Ventilation-Wv78jVhSFTI.mp4" 
        USE_MASK_RCNN = True
        # hacky enum
        X = 0
        Y = 1
        W = 2
        H = 3
    
        if USE_MASK_RCNN:
            rect = self.loadMask()
        else:
            rect = list(cv2.selectROI(imageToProcess))
        
        if rect[W] > rect[H]:
            # subtract half the difference between w and h to keep it centered
            rect[Y] -= (rect[W] - rect[H]) / 2.0
        else: 
            rect[X] -= (rect[H] - rect[W]) / 2.0
        
        max_size = max(rect[W], rect[H])
        rect[W] = max_size
        rect[H] = max_size
        
        # visualize for sanity checking
        display = np.copy(imageToProcess)
        tl = (int(rect[X]), int(rect[Y]))
        br = (int(rect[X] + rect[W]), int(rect[Y] + rect[H]))
        black = (0, 0, 0)
        width = 5
        cv2.rectangle(display, tl, br, black, width)
        cv2.imshow("", display)
        cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        return rect, imageToProcess


keypointExtractor().computeKeypoints()
