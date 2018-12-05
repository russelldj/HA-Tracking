# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
import matplotlib.cm as cm
from scipy.optimize import linear_sum_assignment
import copy

from DaSiamRPN.code.net import SiamRPNvot, SiamRPNBIG
from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
from DaSiamRPN.code.utils import get_axis_aligned_bbox, cxy_wh_2_rect

import mask

# load net
#net = SiamRPNvot()
#net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))

net = SiamRPNBIG()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNBIG.model')))

net.eval().cuda()

class Track():
    def __init__(self, ID, contour):
        self.ID = ID
        self.contour = contour

# image and init box
image_files = sorted(glob.glob('./bag/*.jpg'))
#init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
#[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
ltwh = [246, 177, 167, 133]
[cx, cy, w, h] = [661, 637, 330, 330]#[2 * ltwh[0] + ltwh[2] , 2 * ltwh[1] + 2 * ltwh[3], 2 * ltwh[2], 2 * ltwh[3]]

# tracker init
VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Injection_Preparation.mp4" 
video_reader = cv2.VideoCapture(VIDEO_FILE)
START_FRAME = 240
video_reader.set(1, START_FRAME)
ok, frame = video_reader.read()

ltwh = cv2.selectROI(frame)
print(ltwh)
#ltwh = (726, 445, 29, 73)
cv2.destroyAllWindows()
[cx, cy, w, h] = [ltwh[0] + ltwh[2] / 2 , ltwh[1] + ltwh[3] / 2, ltwh[2], ltwh[3]]
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
state = SiamRPN_init(frame, target_pos, target_sz, net)

H5_FILE = "/home/drussel1/data/EIMP/EIMP_mask-RCNN_detections/Injection_Preparation.mp4.h5"

# tracking and visualization
toc = 0
frame_num = START_FRAME
score = 1
next_ID = 0

tracks = []

while ok:
    contours = mask.extract_masks_one_frame(H5_FILE, frame_num)
     
    cost = np.zeros((len(tracks), len(contours)))
    for i, track in enumerate(tracks):
        for j, contour in enumerate(contours):
            print("track len: {}, contour len: {}, track.ID: {}".format(len(track.contour[0]), len(contour[0]), track.ID))
            cost[i, j] = 1 - mask.slow_mask_IOU(track.contour, contour) # Posed as a cost
    print(cost)
    row_inds, col_inds = linear_sum_assignment(cost)

    # assign ind to masks
    assigned = np.zeros((len(contours),))
    new_tracks = []
    for assigniment_ind, (row_ind, col_ind) in enumerate(zip(row_inds, col_inds)):
        print("cost: {}".format(cost[row_ind, col_ind]))
        tracks[row_ind].contour = contours[col_ind]
        assigned[col_ind] = 1

    current_ids = [track.ID for track in new_tracks]
    print("current ids : {}".format(current_ids))
    if len(tracks) < 2: #TODO improve the logic here
        for ind, val in enumerate(assigned):
            if val == 0:
                new_tracks.append(Track(next_ID, contours[ind]))
                next_ID += 1

    tracks += new_tracks

    if len(contours) > 0:
        mask.slow_mask_IOU(contours[0], contours[0]) 
    vis = mask.draw_mask(frame, [track.contour for track in tracks], [track.ID for track in tracks])
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    color = cm.hot(score)
    color = (color[0] * 255, color[1] * 255, color[2] * 255)
    print(color)
    cv2.rectangle(vis, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), color , 3)
    cv2.putText(vis,str(score), (res[0], res[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    cv2.imshow('SiamRPN', vis)
    cv2.waitKey(1)
    ok, frame = video_reader.read()
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, frame)  # track
    score = state["score"]
    print(score)
    toc += cv2.getTickCount()-tic
    frame_num += 1

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
