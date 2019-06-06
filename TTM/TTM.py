import tools
import sys
import argparse
import cv2
import pdb
import numpy as np
import argparse

from tools import tools, mask, KeypointCapture, KeypointVisualization

"""
THis is going to do everything
Most operations that I want to run will be functions in here which can just be called directly with or without args

"""
VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Injection_Preparation.mp4"
KEYPOINTS_FILE = "./data/TTM-data/processed/EIMP/Mask-Guided-Keypoints-Zero-Padded/Injection_Preparation"
START_FRAME = 300

def computeOpenPoseKeypoints():
    sys.path.append('libs/openpose/build/python');    
    from openpose import pyopenpose as op
    print(op)

def test_H5_load():
    pass

def FullTracking():
    from libs.DaSiamRPN.code import SiamRPN_tracker
    print(SiamRPN_tracker)

def loadKeypoints(foldername=KEYPOINTS_FILE):
    keypoint_capture = KeypointCapture.Read2DJsonPath(foldername, 0, 0)
    return keypoint_capture

def testDaSiamTracking(video_fname=VIDEO_FILE):
    # import the tracker
    from libs.DaSiamRPN.code import SiamRPN_tracker# this takes a while

    parser = argparse.ArgumentParser()
    parser.add_argument("--imshow", help="show the tracking", action="store_true")
    parser.add_argument("--set-search", help="use hand context", action="store_true")
    parser.add_argument("--select-region", help="initialize the tracker by hand", action="store_true")
    parser.add_argument("--video-file", help="The video to run on", default=VIDEO_FILE)
    parser.add_argument("--keypoints-file", help="The folder of keypoints to use", default=KEYPOINTS_FILE)
    parser.add_argument("--start-frame", help="what frame to start at", default=START_FRAME, type=int)
    args = parser.parse_args()
    LOST_THRESH = 0.8
    FINGER_CONF = 0.2#0.2 # still needs to be further tuned
    #FINGER = "Right_Index4"
    SELECT_REGION = args.select_region#False # choose your own initial region
    SET_SEARCH = args.set_search#False # specify where to look
    IMSHOW = args.imshow#False
    OUTPUT_FILENAME = "video_setsearch_{}.avi".format(SET_SEARCH)
    FPS = 30
    WIDTH = 1280
    HEIGHT = 720
    use_hand_box = False
    score = 1
    toc = 0
    frame_num = args.start_frame
    
    # create the visualizer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
    video_reader = cv2.VideoCapture(args.video_file)
    video_reader.set(1, frame_num)
    ok, frame = video_reader.read()
    
    # choose the initial region
    if SELECT_REGION:
        ltwh = list(cv2.selectROI(frame))
    else:
        ltwh = [813, 459, 43, 109]
    print(ltwh)
    hand_box = [float(i) for i in ltwh]
    track_1_diff_xy = [0.0, 0.0]
    cv2.destroyAllWindows()
    # create a wrapper around the tracker
    tracker = SiamRPN_tracker.SiamRPN_tracker(frame, ltwh) 
    tracker.setSearchRegion(ltwh)

    if SET_SEARCH:
        keypoint_capture = loadKeypoints() # the issue here is that there is that the jsons are not zero padded
        visualizer = KeypointVisualization.KeypointVisualization(keypoint_capture)
        i = START_FRAME

        was_lost = False # was it lost in the last frame?
        offset = None

        while ok:
            current_keypoints = keypoint_capture.GetFrameKeypointsAsOneDict(i)
            if tracker.isLost() and not was_lost: # the tracker is lost
                #find the nearest point by getting the location
                tracker_location = tracker.getLocation()
                # doing this iteratives just to avoid errors
                shortest_dist = np.inf
                best_joint = None
                best_location = None
                for joint, (x, y, conf) in current_keypoints.items():
                    if conf < FINGER_CONF:
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

            elif tracker.isLost() and was_lost:
                # update the search region
                # find the best key in the dictionary
                if best_joint in current_keypoints: # this could be none or otherwise not present
                    x_y_conf = current_keypoints[best_joint] # the keypoint we were tracking
                    if x_y_conf[2] > FINGER_CONF: 
                        x_y = x_y_conf[:2]
                        x_y += offset # validate this is the right direction
                        #cv2.circle(frame, tuple([int(x) for x in x_y]), 15, (0,0,0), 10)
                        tracker.setSearchLocation(x_y)

            elif not tracker.isLost() and not was_lost:
                offset = None
                was_lost = False
                best_key = None
                best_location = None
            elif not tracker.isLost() and was_lost:
                offset = None
                was_lost = False
                best_key = None
                best_location = None

            # this should really be as follows:
            # Check if the tracker is lost
            # if it is, find the nearest point
            # track that point
            # that is, find the point in the next frame which is closer to the location of the ones which have that id  
            #if FINGER in current_keypoints and current_keypoints[FINGER][2] != 0:
            #    right_index = current_keypoints[FINGER]
            #    #print("the confs are {}".format([x[2] for x in current_keypoints.values()]))
            #    # this threshold needs to be tuned
            #    if right_index[2] > FINGER_CONF and tracker.isLost():
            #        print("right index {}".format(right_index))
            #        tracker.setSearchLocation(right_index[0:2])
            #        cv2.circle(frame, tuple([int(x) for x in right_index[0:2]]), 5, (255,0,0) , 10)

            cv2.rectangle(frame, (ltwh[0], ltwh[1]), (ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]), (255,0,0) , 3)
            cv2.putText(frame, "conf: {:03f}".format(score), (ltwh[0], ltwh[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            # add the keypoints
            #frame = visualizer.PlotSingleFrameFromIndOpenCV(frame, i)
            frame = visualizer.PlotSingleFrameFromIndOpenCVOpenPose(frame, i)
            if IMSHOW:
                cv2.imshow('SiamRPN', frame)
                cv2.imwrite("frame.png",frame)
            video_writer.write(frame)
            cv2.waitKey(1)
            ok, frame = video_reader.read()
            if not ok:
                break
            tic = cv2.getTickCount()
            ltwh, score, crop_region = tracker.predict(frame)  # track
            crop_region = [int(c) for c in crop_region]
            cv2.rectangle(frame, (crop_region[0], crop_region[1]), (crop_region[2], crop_region[3]), (0,0,255) , 3)
            i += 1

    else:
        while ok:
            cv2.rectangle(frame, (ltwh[0], ltwh[1]), (ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]), (255,0,0) , 3)
            cv2.putText(frame, "conf: {:03f}".format(score), (ltwh[0], ltwh[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            if IMSHOW:
                cv2.imshow('SiamRPN', frame)
                cv2.imwrite("frame.png",frame)
            video_writer.write(frame)
            cv2.waitKey(1)
            ok, frame = video_reader.read()
            if not ok:
                break
            tic = cv2.getTickCount()
            ltwh, score, crop_region = tracker.predict(frame)  # track
            crop_region = [int(c) for c in crop_region]
            cv2.rectangle(frame, (crop_region[0], crop_region[1]), (crop_region[2], crop_region[3]), (0,0,255) , 3)
            cv2.putText(frame, "search window", (crop_region[0], crop_region[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            print(score)
            toc += cv2.getTickCount()-tic
            frame_num += 1


if __name__ == "__main__":
    #loadKeypoints()
    testDaSiamTracking()
    #FullTracking()
    #computeOpenPoseKeypoints()

