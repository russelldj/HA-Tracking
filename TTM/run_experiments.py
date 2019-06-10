import pandas as pd
import cv2  
import numpy as np
import os
import pdb
import threading
from tools import tools, KeypointVisualization, KeypointCapture
from libs.DaSiamRPN.code import SiamRPN_tracker# this takes a while
import glob
import argparse

VIDEO_FILE = 'ADL_videos/P_{:02d}.MP4'
OUTPUT_FOLDER = 'ADL_Market_format/all_bounding_boxes' # TODO check this 
IMG_SHAPE = (128, 128)

class DaSiamShiftSearch():
    def __init__(self):
        self.tracker = None
        self.was_lost = False
        self.last_keypoint = None
        self.FINGER_CONF = 0.4 # TODO choose this more appropriately
        self.DIST_THRESHOLD = 50 # TODO choose this more appropriately
        self.visualizer = KeypointVisualization.KeypointVisualization(KeypointCapture.KeypointCapture())

    def convert_numpy(self, numpy_data):
        ORDERED_KEYPOINTS_HAND =[
                  "Wrist",
                  "Thumb1",
                    "Thumb2",
                      "Thumb3",
                        "Thumb4",
                          "Index1",
                            "Index2",
                              "Index3",
                                "Index4",
                                  "Middle1",
                                    "Middle2",
                                      "Middle3",
                                        "Middle4",
                                          "Ring1",
                                            "Ring2",
                                              "Ring3",
                                                "Ring4",
                                                  "Pinky1",
                                                    "Pinky2",
                                                      "Pinky3",
                                                        "Pinky4"
                                                        ]
        HANDS = ["Left", "Right"]
        assert(numpy_data.shape == (2, 1, 21, 3))
        output = dict()
        for which_hand, hand in enumerate(numpy_data):
            for which_finger, finger in enumerate(hand[0]): #weird extra dimension
                ID = "{}_{}".format(HANDS[which_hand], ORDERED_KEYPOINTS_HAND[which_finger])
                output[ID] = finger
        return output
    
    def move_region(self, current_keypoints):
        if self.tracker.isLost() and not self.was_lost: # the tracker is lost for the first time
            #find the nearest point by getting the location
            tracker_location = self.tracker.getLocation() # TODO make sure that this is the center of the output
            # doing this iteratives just to avoid errors
            self.best_joint = None 
            shortest_dist = self.DIST_THRESHOLD # TODO threshold on this
            best_location = None
            for joint, (x, y, conf) in current_keypoints.items():
                if conf < self.FINGER_CONF:
                    continue # just skip this one
                keypoint_location = np.array([x, y])
                print("key: {}, value {}".format(joint, (x,y,conf)))
                dist = np.linalg.norm(tracker_location - keypoint_location)
                if dist < shortest_dist:
                    self.best_joint = joint
                    best_location = keypoint_location
                    shortest_dist = dist
                    self.offset = tracker_location - best_location # TODO determine if this is what we really want
    
            if best_location is not None: # check if this was actually set
                #cv2.circle(frame, tuple([int(x) for x in best_location.tolist()]), 15, (255,255,255), 10)
                self.was_lost = True # only do this if you find a keypoint
    
        elif self.tracker.isLost() and self.was_lost: # TODO if it's not there, find another one which is the closest
            # update the search region
            # find the best key in the dictionary
            if self.best_joint in current_keypoints: # this could be none or otherwise not present
                x_y_conf = current_keypoints[self.best_joint] # the keypoint we were tracking
                if x_y_conf[2] > self.FINGER_CONF: 
                    x_y = x_y_conf[:2]
                    x_y += self.offset # validate this is the right direction
                    self.tracker.setSearchLocation(x_y)
                    # maybe shift it differently if the nearest on isn't present


    
        elif not self.tracker.isLost() and not self.was_lost:
            self.offset = None
            self.was_lost = False
            self.best_key = None
            self.best_location = None
        elif not self.tracker.isLost() and self.was_lost:
            self.offset = None
            self.was_lost = False
            self.best_key = None
            self.best_location = None


    def visualize(self, frame, ltwh, score, keypoint_dict):
        if IMSHOW:
            frame = self.visualizer.PlotSingleFrameFromAndKeypointDict(frame, keypoint_dict, self.FINGER_CONF)
            cv2.rectangle(frame, (ltwh[0], ltwh[1]), (ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]), (255,0,0) , 3)
            cv2.putText(frame, "conf: {:03f}".format(score), (ltwh[0], ltwh[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.imshow("predicted location", frame)
            cv2.waitKey(1)
            


    def track_section(self, track, vid, outfile, set_search=False, numpy_files=None):
        initial_bbox = track.iloc[0][[1,2,3,4]].tolist()
        initial_bbox = tools.ltrb_to_tlbr(initial_bbox)
        initial_bbox = tools.tlbr_to_ltwh(initial_bbox) # both are needed
        initial_bbox *= 2 #ADL annotations are off by two I think
        index = track.iloc[0]["frame"]
        obj_class = track.iloc[0]["class"]
        obj_ID = track.iloc[0]["ID"]
    
        final_index = track.iloc[-1]["frame"]
    
        vid.set(1,index)
        print(index)
        print(vid.get(1))
        ok, frame = vid.read()
        self.tracker = SiamRPN_tracker.SiamRPN_tracker(frame, initial_bbox)
    
        while index <= final_index and ok:# the current frame 
            #this is where I want to update it to grab the numpy frame
            if set_search and index < len(numpy_files):
                print(numpy_files[index])
                keypoints = np.load(numpy_files[index])
                keypoints_dict = self.convert_numpy(keypoints)
                self.move_region(keypoints_dict) # TODO check if this is working
            else:
                keypoints_dict = {}
                if set_search: # clean up
                    print("Missing numpy file, not setting the search region")
    
            ltwh, score, crop_region = self.tracker.predict(frame)
            self.visualize(frame, ltwh, score, keypoints_dict)
            WRITE_ADL=False
            if WRITE_ADL:
                tlbr = tools.ltwh_to_tlbr(ltwh)
                l, t, r, b = tools.tlbr_to_ltrb(tlbr)
                outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(obj_ID, l, t, r, b, index, 0, obj_class)) # TODO validate this line
            else:
                x, y, w, h = ltwh # TODO check more
                #'FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
                outfile.write("{} {} {} {} {} {} {} {} 1\n".format(index, obj_ID, x, y, w, h, 1, 1)) # TODO add the class id

            ok, frame = vid.read()
            if frame is None:
                break # this is just a messier way of doing the while check 
            index+=1
    
    def run_video(self, start_vid, end_vid, output_folder):# inclusive, exclusive
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
        if SET_SHIFT:
            output_folder = os.path.join(output_folder, "{}_{}_{}".format(SET_SHIFT, self.FINGER_CONF, self.DIST_THRESHOLD))
        if not os.path.isdir(output_folder):
            os.system("mkdir -p {}".format(output_folder))

        for i in range(start_vid, end_vid):
            numpy_folder = '/home/drussel1/data/ADL/openpose_keypoints/keypoint_{:02d}/*.npy'.format(i)
            numpy_files = sorted(glob.glob(numpy_folder))
            df = pd.read_csv('/home/drussel1/data/ADL/ADL_annotations/object_annotation/object_annot_P_{:02d}.txt'.format(i), sep=' ', 
                        names=['ID', 'x1', 'y1', 'x2', 'y2', 'frame', 'active',
                        'class', 'NaN'], index_col=None)

            vid = cv2.VideoCapture('/home/drussel1/data/ADL/ADL_videos/P_{:02d}.MP4'.format(i))
            with open(os.path.join(output_folder, "P_{:02d}.txt".format(i)), "w") as outfile:#TODO pass in as parameter
            #with open("test_track.csv", "w") as outfile:
                #pdb.set_trace()
                df.sort_values(by=['ID', 'frame'], inplace = True)
                 
                IDs = list(set(df['ID'].tolist()))
                for ID in IDs:
                    track = df.loc[df['ID'] == ID]

                    self.track_section(track, vid, outfile, SET_SHIFT, numpy_files)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__": 
    # creating thread 
    OUTPUT_FOLDER = "/usr0/home/drussel1/dev/TTM/TTM/data/CVPR/6_9_runs"
    #OUTPUT_FOLDER = "/usr0/home/drussel1/dev/TTM/TTM/data/CVPR/temp"
    SET_SHIFT = True
    IMSHOW = False
    OUTPUT_FILENAME = "test.avi"
    FPS = 30
    (WIDTH, HEIGHT) = (1280,960)
    args = parse_args()

    DaSiamShiftSearch().run_video(args.start,args.stop, OUTPUT_FOLDER)
    #DaSiamShiftSearch().run_video(5,10, OUTPUT_FOLDER)
    #DaSiamShiftSearch().run_video(10,15, OUTPUT_FOLDER)
    #DaSiamShiftSearch().run_video(15,21, OUTPUT_FOLDER)

