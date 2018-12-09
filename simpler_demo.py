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

if True:
    VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Injection_Preparation.mp4" 
    H5_FILE = "/home/drussel1/data/EIMP/EIMP_mask-RCNN_detections/Injection_Preparation.mp4.h5"
    START_FRAME = 300
if False:
    VIDEO_FILE = "/home/drussel1/data/ADL/ADL1619_videos/P_18.MP4"
    H5_FILE = "/home/drussel1/data/ADL/new_mask_outputs/dataset_per_frame/P_18.MP4.h5" 
    START_FRAME = 18000 
if False:
    VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Oxygen_Saturation-YzkxXmT4neg.mp4"
    H5_FILE = "/home/drussel1/data/EIMP/new-EIMP-mask-RCNN-detections/Oxygen_Saturation-YzkxXmT4neg.mp4.h5" 
    START_FRAME = 300


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
ltwh = list(cv2.selectROI(frame))
hand_box = [float(i) for i in ltwh]
track_1_diff_xy = [0.0, 0.0]
cv2.destroyAllWindows()
tracker = SiamRPN_tracker(frame, ltwh) 
# tracking and visualization
toc = 0
frame_num = START_FRAME
score = 1
next_ID = 0

os.system("/bin/rm chips/*")

while ok:
    cv2.rectangle(frame, (ltwh[0], ltwh[1]), (ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]), (255,0,0) , 3)
    cv2.putText(frame, str(score), (ltwh[0], ltwh[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow('SiamRPN', frame)
    video_writer.write(frame)
    cv2.waitKey(1)
    ok, frame = video_reader.read()
    tic = cv2.getTickCount()
    ltwh, score = tracker.predict(frame)  # track
    print(score)
    toc += cv2.getTickCount()-tic
    frame_num += 1

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
