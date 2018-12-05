# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from DaSiamRPN.code.net import SiamRPNvot
from DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
from DaSiamRPN.code.utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
net.eval().cuda()

# image and init box
image_files = sorted(glob.glob('./bag/*.jpg'))
#init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
#[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
ltwh = [246, 177, 167, 133]
[cx, cy, w, h] = [661, 637, 330, 330]#[2 * ltwh[0] + ltwh[2] , 2 * ltwh[1] + 2 * ltwh[3], 2 * ltwh[2], 2 * ltwh[3]]


# tracker init
VIDEO_FILE = "/home/drussel1/data/EIMP/videos/Injection_Preparation.mp4" 
video_reader = cv2.VideoCapture(VIDEO_FILE)
video_reader.set(1, 240)
ok, frame = video_reader.read()

ltwh = cv2.selectROI(frame)
cv2.destroyAllWindows()
[cx, cy, w, h] = [ltwh[0] + ltwh[2] / 2 , ltwh[1] + ltwh[3] / 2, ltwh[2], ltwh[3]]
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
state = SiamRPN_init(frame, target_pos, target_sz, net)

# tracking and visualization
toc = 0
while ok:
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', frame)
    cv2.waitKey(1)
    ok, frame = video_reader.read()
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, frame)  # track
    toc += cv2.getTickCount()-tic

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
