import pandas as pd
import cv2  
import numpy as np
import os
import pdb
import threading
from tools import tools
from libs.DaSiamRPN.code import SiamRPN_tracker# this takes a while
import glob

VIDEO_FILE = 'ADL_videos/P_{:02d}.MP4'
OUTPUT_FOLDER = 'ADL_Market_format/all_bounding_boxes'
IMG_SHAPE = (128, 128)

class DaSiamShiftSearch():
    def __init__(self):
        self.tracker = None
        self.was_lost = False
        self.last_keypoint = None
        self.FINGER_CONF = 0.7

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
        HANDS = ["left", "right"]
        assert(numpy_data.shape == (2, 1, 21, 3))
        output = dict()
        for which_hand, hand in enumerate(numpy_data):
            for which_finger, finger in enumerate(hand[0]): #weird extra dimension
                ID = "{}_{}".format(HANDS[which_hand], ORDERED_KEYPOINTS_HAND[which_finger])
                output[ID] = finger
        return output
    
    def move_region(self, current_keypoints):
        if self.tracker.isLost() and not self.was_lost: # the tracker is lost
            #find the nearest point by getting the location
            tracker_location = self.tracker.getLocation()
            # doing this iteratives just to avoid errors
            shortest_dist = np.inf
            best_joint = None
            best_location = None
            for joint, (x, y, conf) in current_keypoints.items():
                if conf < self.FINGER_CONF:
                    continue # just skip this one
                keypoint_location = np.array([x, y])
                print("key: {}, value {}".format(joint, (x,y,conf)))
                dist = np.linalg.norm(tracker_location - keypoint_location)
                if dist < shortest_dist:
                    best_joint = joint
                    best_location = keypoint_location
                    shortest_dist = dist
                offset = tracker_location - best_location
    
            if best_location is not None: # check if this was actually set
                pass
                #cv2.circle(frame, tuple([int(x) for x in best_location.tolist()]), 15, (255,255,255), 10)
            was_lost = True
    
        elif self.tracker.isLost() and self.was_lost:
            # update the search region
            # find the best key in the dictionary
            if best_joint in current_keypoints: # this could be none or otherwise not present
                x_y_conf = current_keypoints[best_joint] # the keypoint we were tracking
                if x_y_conf[2] > FINGER_CONF: 
                    x_y = x_y_conf[:2]
                    x_y += self.offset # validate this is the right direction
                    #cv2.circle(frame, tuple([int(x) for x in x_y]), 15, (0,0,0), 10)
                    self.tracker.setSearchLocation(x_y)
    
        elif not self.tracker.isLost() and not self.was_lost:
            self.offset = None
            self.was_lost = False
            self.best_key = None
            self.best_location = None
        elif not self.tracker.isLost() and was_lost:
            self.offset = None
            self.was_lost = False
            self.best_key = None
            self.best_location = None
   

    def track_section(self, track, vid, outfile, set_search=False, numpy_files=None):
        initial_bbox = track.iloc[0][[1,2,3,4]].tolist()
        initial_bbox = tools.ltrb_to_tlbr(initial_bbox)
        initial_bbox = tools.tlbr_to_ltwh(initial_bbox) # both are needed
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
            if set_search:
                assert(numpy_files is not None)
                print(numpy_files[index])
                keypoints = np.load(numpy_files[index])
                keypoints_dict = self.convert_numpy(keypoints)

            self.move_region(keypoints_dict) # TODO check if this is working
    
            ltwh, score, crop_region = self.tracker.predict(frame)
            tlbr = tools.ltwh_to_tlbr(ltwh)
            l, t, r, b = tools.tlbr_to_ltrb(tlbr)
            outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(obj_ID, l, t, r, b, int(vid.get(1)), 0, obj_class)) # TODO validate this line
            ok, frame = vid.read()
            if frame is None:
                break # this is just a messier way of doing the while check 
            index+=1
    
    def run_video(self, start_vid, end_vid):# inclusive, exclusive
        for i in range(start_vid, end_vid):
            numpy_folder = '/home/drussel1/data/ADL/openpose_keypoints/keypoint_{:02d}/*.npy'.format(i)
            numpy_files = sorted(glob.glob(numpy_folder))
            df = pd.read_csv('/home/drussel1/data/ADL/ADL_annotations/object_annotation/object_annot_P_{:02d}.txt'.format(i), sep=' ', 
                        names=['ID', 'x1', 'y1', 'x2', 'y2', 'frame', 'active',
                        'class', 'NaN'], index_col=None)
            vid = cv2.VideoCapture('/home/drussel1/data/ADL/ADL_videos/P_{:02d}.MP4'.format(i))
            with open("/home/drussel1/dev/TTM/TTM/outputs/P_{:02d}.txt".format(i), "w") as outfile:
            #with open("test_track.csv", "w") as outfile:
                #pdb.set_trace()
                df.sort_values(by=['ID', 'frame'], inplace = True)
                 
                IDs = list(set(df['ID'].tolist()))
                for ID in IDs:
                    track = df.loc[df['ID'] == ID]
                    self.track_section(track, vid, outfile, True, numpy_files)
    
    
if __name__ == "__main__": 
    # creating thread 
    #DaSiamShiftSearch().run_video(1,6)
    #DaSiamShiftSearch().run_video(6,11)
    #DaSiamShiftSearch().run_video(11,16)
    DaSiamShiftSearch().run_video(16,21)
    #run_video(6,11)
    #run_video(11,16)
    #run_video(16,21)
    #for i in range(3,6,2):
    #    all_threads.append(threading.Thread(target=run_video, args=(i,i+2)))

    #for a_thread in all_threads:
    #    a_thread.start()

    #for a_thread in all_threads:
    #    a_thread.join()

  
    #for index, row in df.iterrows():
    #       print(row['frame'])
    #       cap.set(1, row['frame'])
    #       ret, img = cap.read()
    #       if not ret:
    #           break
    #       crop = img[2*row['y1']:2*row['y2'],2*row['x1']:2*row['x2']].copy()
    #       crop = cv2.resize(crop, IMG_SHAPE, cv2.INTER_CUBIC)  
    #       print(crop.shape)
    #       cv2.imwrite('{}/{:02d}{:04d}_c1s1_{}_00.jpg'.format(OUTPUT_FOLDER, i, row['ID'], row['frame']), crop)
