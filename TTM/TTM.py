import tools
import sys
import argparse
import cv2
import pdb

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
    from libs.DaSiamRPN.code.SiamRPN_tracker import SiamRPN_tracker
    print(SiamRPN_tracker)

def loadKeypoints(foldername=KEYPOINTS_FILE):
    keypoint_capture = KeypointCapture.Read2DJsonPath(foldername, 0, 0)
    return keypoint_capture

def testDaSiamTracking(video_fname=VIDEO_FILE):
    # import the tracker
    from libs.DaSiamRPN.code.SiamRPN_tracker import SiamRPN_tracker
    LOST_THRESH = 0.8
    FINGER_CONF = 0.2#0.2 # still needs to be further tuned
    FINGER = "Right_Index4"
    OUTPUT_FILENAME = "video.avi"
    SELECT_REGION = False # choose your own initial region
    SET_SEARCH = True # specify where to look
    FPS = 30
    WIDTH = 1280
    HEIGHT = 720
    use_hand_box = False
    score = 1
    toc = 0
    frame_num = START_FRAME
    
    # create the visualizer
    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
    video_reader = cv2.VideoCapture(VIDEO_FILE)
    video_reader.set(1, START_FRAME)
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
    tracker = SiamRPN_tracker(frame, ltwh) 
    tracker.setSearchRegion(ltwh)

    if SET_SEARCH:
        keypoint_capture = loadKeypoints() # the issue here is that there is that the jsons are not zero padded
        visualizer = KeypointVisualization.KeypointVisualization(keypoint_capture)
        i = START_FRAME
        while ok:
            current_keypoints = keypoint_capture.GetFrameKeypointsAsOneDict(i)
            if FINGER in current_keypoints and current_keypoints[FINGER][2] != 0:
                right_index = current_keypoints[FINGER]
                #print("the confs are {}".format([x[2] for x in current_keypoints.values()]))
                # this threshold needs to be tuned
                if right_index[2] > FINGER_CONF and tracker.isLost():
                    print("right index {}".format(right_index))
                    tracker.setSearchLocation(right_index[0:2])
                    cv2.circle(frame, tuple([int(x) for x in right_index[0:2]]), 5, (255,0,0) , 10)

            cv2.rectangle(frame, (ltwh[0], ltwh[1]), (ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]), (255,0,0) , 3)
            cv2.putText(frame, "conf: {:03f}".format(score), (ltwh[0], ltwh[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            # add the keypoints
            frame = visualizer.PlotSingleFrameFromIndOpenCV(frame, i)
            cv2.imshow('SiamRPN', frame)
            video_writer.write(frame)
            cv2.waitKey(1)
            ok, frame = video_reader.read()
            tic = cv2.getTickCount()
            #tracker.setSearchRegion([100, 100, 50, 100]) # set search region
            ltwh, score, crop_region = tracker.predict(frame)  # track
            crop_region = [int(c) for c in crop_region]
            cv2.rectangle(frame, (crop_region[0], crop_region[1]), (crop_region[2], crop_region[3]), (0,0,255) , 3)
            i += 1

    else:
        while ok:
            cv2.rectangle(frame, (ltwh[0], ltwh[1]), (ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]), (255,0,0) , 3)
            cv2.putText(frame, "conf: {:03f}".format(score), (ltwh[0], ltwh[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.imshow('SiamRPN', frame)
            video_writer.write(frame)
            cv2.waitKey(1)
            ok, frame = video_reader.read()
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

