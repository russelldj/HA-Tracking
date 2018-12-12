import numpy as np
import cv2
import tools
#from pycocotools import mask as MaskUtil
import h5py
import json
import pdb
from scipy.optimize import linear_sum_assignment

from .track import Track

"""
this is a collection of helper functions which help with manipulating maskss
"""

def bbox_to_contour(bbox):
    """takes in a bounding box in [ltwh] format and outputs the associated mask in the openCV format

    params
    ----------  
    bbox : arrayLike

    returns
    ---------- 
    mask : list(List(List(List(int))))
        the first list is the number of discontinous contours, then the next level is points as [[x,y]]
    """
    assert len(bbox) == 4
    tlbr = tools.ltwh_to_tlbr(bbox)
    top, left, bottom, right = [int(x) for x in tlbr]
    assert bottom > top and right > left
    contour = [[ [[left, bottom]], [[right, bottom]], [[right, top]], [[left, top]]]]
    return contour
    
def boundary_to_RLE(image, contours):
    mask = cv2.fillPoly(np.zeros(image.shape[:2]), pts=[np.asarray(c) for c in contours], color=(255,255,255))
    mask = np.asfortranarray(mask).astype(np.uint8)  # needed for the encoding step
    cv2.imwrite("test.png", mask)
    mask = MaskUtil.encode(mask)
    return mask

def extract_masks_one_frame(h5_filename, ind, threshold=0.5, target_class = 1):
    """
    >>> extract_masks_one_frame("/home/drussel1/data/EIMP/EIMP_mask-RCNN_detections/Injection_Preparation.mp4.h5", 200)
    """
    h5_file = h5py.File(h5_filename, "r")
    ind = str(ind)
    assert ind in h5_file.keys(), "ind: {} wasn't in the keys for {}".format(ind, h5_filename)
    data = json.loads(h5_file[ind].value)
    assert list(data.keys()) == ['frame', 'classes', 'video', 'boxes', 'contours']
    contours = data["contours"]
    boxes    = data["boxes"]
    classes  = data["classes"]

    # first filtering op
    output = zip(*((box[4], contour, clas) for box, contour, clas in zip(boxes, contours, classes) if box[4] > threshold)) # box[4] is the conf 
    list_output = list(output)
    if len(list_output) == 3:
        confs, contours, classes = list_output
    else: 
        confs, contours, classes = [], [], []
    
    # and second
    output = zip(*[(conf, contour, clas) for conf, contour, clas in zip(confs, contours, classes) if (target_class is None or clas==target_class)]) # box[4] is the conf 

    list_output = list(output)
    if len(list_output) == 3:
        confs, contours, classes = list_output
    else: 
        confs, contours, classes = [], [], []
    contours = list(contours)

    assert type(contours) == list, type(contours)
    return contours

def contour_to_biggest_mask(image_shape, contours):
    pts = [[np.asarray(c_, dtype = int) for c_ in c] for c in contours] 
    mask  = np.zeros(image_shape)
    for i, pt in enumerate(pts):
        for sec in pt:
            temp_mask = cv2.fillPoly(np.zeros(image_shape), pts=[sec], color=1)
            if sum(sum(temp_mask)) > sum(sum(mask)):
                mask = temp_mask
    return mask

def draw_mask(image, contours, IDs, color=(0,0,0)):
    """
    >>> contours = extract_masks_one_frame("/home/drussel1/data/EIMP/EIMP_mask-RCNN_detections/Injection_Preparation.mp4.h5", 1)
    ... import cv2
    ... import numpy as np
    ... img = cv2.imread("/home/drussel1/data/EIMP/frames/000001.jpeg") 
    ... draw_mask(img, contours)
    """
    assert len(IDs) == len(contours)
    pts = [[np.asarray(c_, dtype = int) for c_ in c] for c in contours] 
    overlay_image = np.zeros_like(image)
    np.random.seed(0)
    colors = [tuple(np.random.randint(0, 255, (3,)).tolist()) for _ in range(10000)]

    for i, pt in enumerate(pts):
        overlay_image = cv2.fillPoly(overlay_image, pts=pt, color=colors[IDs[i]])
    
    image = cv2.addWeighted(image,0.7,overlay_image,0.3,0)
    return image

def slow_mask_IOU(contour1, contour2, image_shape=(720,1280)):
    contour1 = [np.asarray(c_, dtype = int) for c_ in contour1]
    contour2 = [np.asarray(c_, dtype = int) for c_ in contour2]
    
    mask1 = cv2.fillPoly(np.zeros((image_shape)), pts=contour1, color=1)
    mask2 = cv2.fillPoly(np.zeros((image_shape)), pts=contour2, color=1)
    overlap = sum(sum(np.multiply(mask1, mask2)))
    total = sum(sum(mask1)) + sum(sum(mask2)) - overlap 
    IOU = float(overlap) / total
    return IOU

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


def match_masks(contours, tracks, frame, next_ID):
    cost = np.zeros((len(tracks), len(contours)))
    for i, track in enumerate(tracks):
        for j, contour in enumerate(contours):
            print("track len: {}, contour len: {}, track.ID: {}".format(len(track.contour[0]), len(contour[0]), track.ID))
            item = 1 - slow_mask_IOU(track.contour, contour, frame.shape[:2]) # Posed as a cost
            if np.isnan(item):
                pdb.set_trace()
                item = 1 - slow_mask_IOU(track.contour, contour) # Posed as a cost
            cost[i, j] = item

    print(cost)
    row_inds, col_inds = linear_sum_assignment(cost)

    # assign ind to masks
    assigned = np.zeros((len(contours),))
    new_tracks = []
    for assigniment_ind, (row_ind, col_ind) in enumerate(zip(row_inds, col_inds)):
        if tracks[row_ind].ID == 1:
            print("cost: {}".format(cost[row_ind, col_ind]))
            old_contour = copy.copy(tracks[row_ind].contour)
            new_contour = copy.copy(contours[col_ind])
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
    return next_ID # the track will be updated by reference
