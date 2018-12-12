# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
import matplotlib.cm as cm
from scipy.optimize import linear_sum_assignment
import copy
from sklearn.decomposition import PCA

from DaSiamRPN.code.net import SiamRPNvot#, SiamRPNBIG
from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
from DaSiamRPN.code.utils import get_axis_aligned_bbox, cxy_wh_2_rect

import tools.mask as mask
import pdb

# load net
#net = SiamRPNvot()
#net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))

net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))

net.eval().cuda()

class Track():
    def __init__(self, ID, contour):
        self.ID = ID
        self.contour = contour
        self.diff = 0

# image and init box
image_files = sorted(glob.glob('./bag/*.jpg'))
#init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
#[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

which_one = 0 

if which_one == 0:
    VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Injection_Preparation.mp4" 
    H5_FILE = "/home/drussel1/data/EIMP/EIMP_mask-RCNN_detections/Injection_Preparation.mp4.h5"
    START_FRAME = 600 
if which_one == 1:
    VIDEO_FILE = "/home/drussel1/data/ADL/ADL1619_videos/P_18.MP4"
    H5_FILE = "/home/drussel1/data/ADL/new_mask_outputs/dataset_per_frame/P_18.MP4.h5" 
    START_FRAME = 18000 
if which_one == 2:
    VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Oxygen_Saturation-YzkxXmT4neg.mp4"
    H5_FILE = "/home/drussel1/data/EIMP/new-EIMP-mask-RCNN-detections/Oxygen_Saturation-YzkxXmT4neg.mp4.h5" 
    START_FRAME = 1


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
[cx, cy, w, h] = [ltwh[0] + ltwh[2] / 2 , ltwh[1] + ltwh[3] / 2, ltwh[2], ltwh[3]]
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
state = SiamRPN_init(frame, target_pos, target_sz, net)

# tracking and visualization
toc = 0
frame_num = START_FRAME
score = 1
next_ID = 0

tracks = []

def compute_mask_translation(first_contour, second_contour, point=None, image_shape=None):
    mask1 = mask.contour_to_biggest_mask(image_shape, [first_contour]) 
    mask2 = mask.contour_to_biggest_mask(image_shape, [second_contour])

    cv2.imwrite("mask1.png", mask1*255)
    cv2.imwrite("mask2.png", mask2*255)

    loc1 = np.nonzero(mask1)
    loc2 = np.nonzero(mask2)
    pca1  = PCA(n_components=2)
    pca1.fit(np.asarray([loc1[1], loc1[0]]).transpose()) # xy format
    pca_point = pca1.transform(point)

    pca2  = PCA(n_components=2)
    pca2.fit(np.asarray([loc2[1], loc2[0]]).transpose()) # xy format
    projected_points = []
    for i in [1, -1]:
        for j in [1, -1]:
            new_point = pca2.inverse_transform(np.asarray( [[pca_point[0, 0] * i, pca_point[0, 1] * j]]) )
            projected_points.append(new_point)

    dists = [np.linalg.norm( point - pp ) for pp in projected_points]
    index = dists.index(min(dists))
    new_point = projected_points[index]
    print(dists, index)

    #new_point = pca2.inverse_transform(pca_point)
    print("old point {}, new point {}".format(point, new_point))
    return new_point[0].tolist() #is is shape (1, 2)

    #ave1 = np.average(loc1, axis=1)
    #ave2 = np.average(loc2, axis=1)

    #diff = ave2 - ave1
    #return diff
    

while ok:
    contours = mask.extract_masks_one_frame(H5_FILE, frame_num)
     
    cost = np.zeros((len(tracks), len(contours)))
    for i, track in enumerate(tracks):
        for j, contour in enumerate(contours):
            print("track len: {}, contour len: {}, track.ID: {}".format(len(track.contour[0]), len(contour[0]), track.ID))
            item = 1 - mask.slow_mask_IOU(track.contour, contour, frame.shape[:2]) # Posed as a cost
            if np.isnan(item):
                pdb.set_trace()
                item = 1 - mask.slow_mask_IOU(track.contour, contour) # Posed as a cost
            cost[i, j] = item 

    print(cost)
    row_inds, col_inds = linear_sum_assignment(cost)

    # assign ind to masks
    assigned = np.zeros((len(contours),))
    new_tracks = []
    for assigniment_ind, (row_ind, col_ind) in enumerate(zip(row_inds, col_inds)):

        if tracks[row_ind].ID == 1:
            use_hand_box = True
            print("cost: {}".format(cost[row_ind, col_ind]))
            old_contour = copy.copy(tracks[row_ind].contour)
            new_contour = copy.copy(contours[col_ind])
            new_loc = compute_mask_translation(old_contour, new_contour, np.asarray([hand_box[:2]]), frame.shape[:2])
            hand_box[0] = new_loc[0]
            hand_box[1] = new_loc[1]
        #    hand_box[0] += diff[1]
        #    hand_box[1] += diff[0]
        #    track_1_diff_xy = [diff[1], diff[0]]
        #tracks[row_ind].diff = diff
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
    if score < LOST_THRESH:
        cv2.rectangle(vis, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 0, 255) , 3)
    else:
        cv2.rectangle(vis, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), color , 3)
    if score > .98:
        hand_box = [float(i) for i in res]


    hb = [int(f) for f in hand_box]
    cv2.rectangle(vis, (hb[0], hb[1]), (hb[0] + hb[2], hb[1] + hb[3]), (255,0,0) , 3)


    cv2.putText(vis,str(score), (res[0], res[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    cv2.imshow('SiamRPN', vis)
    video_writer.write(vis)
    cv2.waitKey(1)
    ok, frame = video_reader.read()
    tic = cv2.getTickCount()

    if use_hand_box and state["score"] < LOST_THRESH and False: # TODO implement something which only uses this if the object is  
        state['target_pos'] = [hand_box[0] + hand_box[2], hand_box[1], hand_box[3]] # this is only different if the confidence is low. It's the center
    #state['target_pos'] = [hand_box[0], hand_box[1]] # this is only different if the confidence is low. It's the center
    #
    state = SiamRPN_track(state, frame)  # track
    score = state["score"]
    print(score)
    toc += cv2.getTickCount()-tic
    frame_num += 1

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
