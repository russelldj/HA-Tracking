# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
import os
import matplotlib.cm as cm
from scipy.optimize import linear_sum_assignment
import copy
from sklearn.decomposition import PCA

from DaSiamRPN.code.SiamRPN_tracker import SiamRPN_tracker

import tools.mask as mask
import pdb

# load net
#net = SiamRPNvot()
#net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))

WHICH_ONE=0

if WHICH_ONE==0:
    VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Injection_Preparation.mp4" 
    H5_FILE = "/home/drussel1/data/EIMP/EIMP_mask-RCNN_detections/Injection_Preparation.mp4.h5"
    START_FRAME = 600
if WHICH_ONE==1:
    VIDEO_FILE = "/home/drussel1/data/ADL/ADL1619_videos/P_18.MP4"
    H5_FILE = "/home/drussel1/data/ADL/new_mask_outputs/dataset_per_frame/P_18.MP4.h5" 
    START_FRAME = 18000 
if WHICH_ONE==2:
    VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Oxygen_Saturation-YzkxXmT4neg.mp4"
    H5_FILE = "/home/drussel1/data/EIMP/new-EIMP-mask-RCNN-detections/Oxygen_Saturation-YzkxXmT4neg.mp4.h5" 
    START_FRAME = 300
if WHICH_ONE==3:
    VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Blood_Glucose-6Q-UJJmTMQA.mp4"
    H5_FILE = "/home/drussel1/data/EIMP/new-EIMP-mask-RCNN-detections/Blood_Glucose-6Q-UJJmTMQA.mp4.h5" 
    START_FRAME = 496

LOST_THRESH = 0.8
OUTPUT_FILENAME = "video.avi" 
FPS = 30
WIDTH = 1280 
HEIGHT = 720
use_hand_box = False

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
video_reader = cv2.VideoCapture(VIDEO_FILE)
video_reader.set(1, START_FRAME)
ok, frame = video_reader.read()

#print(ltwh)
#ltwh = (724, 446, 25, 77)
#ltwh = list(cv2.selectROI(frame))
ltwh = [0, 0, 10, 10]
hand_box = [float(i) for i in ltwh]
track_1_diff_xy = [0.0, 0.0]
cv2.destroyAllWindows()
USE_MOSSE=True
#if USE_MOSSE:
#    tracker = cv2.TrackerMOSSE_create()
#    tracker.init(frame, ltwh)
#else:
#    tracker = SiamRPN_tracker(frame, ltwh) 
# tracking and visualization
toc = 0
frame_num = START_FRAME
score = 1
next_ID = 0
if len(glob.glob("chips/*")) > 0:
    os.system("/bin/rm chips/*")

tracks = []

while ok:
    cv2.rectangle(frame, (ltwh[0], ltwh[1]), (ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]), (255,0,0) , 3)
    cv2.putText(frame, str(score), (ltwh[0], ltwh[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    video_writer.write(frame)
    pdb.set_trace()
    contours = mask.extract_masks_one_frame(H5_FILE, frame_num)
    frame_num += 1
    next_ID = mask.match_masks(contours, tracks, frame, next_ID)
    print("length of tracks is {}".format(len(tracks)))
    frame = mask.draw_mask(frame, [track.contour for track in tracks], [track.ID for track in tracks])
    cv2.imshow('SiamRPN', frame)
    cv2.waitKey(1)
    ok, frame = video_reader.read()
    tic = cv2.getTickCount()
    #if USE_MOSSE:
    #    pdb.set_trace()
    #    ok, ltwh = tracker.update(frame)
    #    score = int(ok)
    #else:
    #    ltwh, score = tracker.predict(frame)  # The order of the outputs will need to get switched
    print(score)
    toc += cv2.getTickCount()-tic
    frame_num += 1
