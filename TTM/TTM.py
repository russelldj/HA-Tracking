import tools
import sys
import argparse
import cv2
import pdb

from tools import tools, mask, KeypointCapture

"""
THis is going to do everything
Most operations that I want to run will be functions in here which can just be called directly with or without args

"""
VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Injection_Preparation.mp4"
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

def loadKeypoints(foldername="./data/TTM-data/processed/EIMP/Mask-Guided-Keypoints/Blood_Glucose-6Q-UJJmTMQA/"):
    keypoint_capture = KeypointCapture.Read2DJsonPath(foldername, 0, 0)
    for i in range(keypoint_capture.num_frames):
        print(keypoint_capture.GetFrameKeypointsAsOneDict(i), i)
    print(keypoint_capture.GetFrameKeypointsAsOneDict(0))

def testDaSiamTracking(video_fname=VIDEO_FILE):
    from libs.DaSiamRPN.code.SiamRPN_tracker import SiamRPN_tracker
    LOST_THRESH = 0.8
    OUTPUT_FILENAME = "video.avi"
    FPS = 30
    WIDTH = 1280
    HEIGHT = 720
    use_hand_box = False
    score = 1
    toc = 0
    frame_num = START_FRAME
    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
    video_reader = cv2.VideoCapture(VIDEO_FILE)
    video_reader.set(1, START_FRAME)
    ok, frame = video_reader.read()
    
    ltwh = list(cv2.selectROI(frame))
    hand_box = [float(i) for i in ltwh]
    track_1_diff_xy = [0.0, 0.0]
    cv2.destroyAllWindows()
    tracker = SiamRPN_tracker(frame, ltwh) 

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
    loadKeypoints()
    #testDaSiamTracking()
    #FullTracking()
    #computeOpenPoseKeypoints()

