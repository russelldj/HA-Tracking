import numpy as np
import cv2
import tools
from pycocotools import mask as MaskUtil
import h5py
import json

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
    print(top, left, bottom, right)
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
    confs, contours, classes = zip(*((box[4], contour, clas) for box, contour, clas in zip(boxes, contours, classes) if box[4] > threshold)) # box[4] is the conf 
    output = zip(*((conf, contour, clas) for conf, contour, clas in zip(confs, contours, classes) if (target_class is None or clas==target_class))) # box[4] is the conf 
    list_output = list(output)
    if len(list_output) == 3:
        confs, contours, classes = list_output
    else: 
        confs, contours, classes = [], [], []
    contours = list(contours)

    print(len(contours))
    assert type(contours) == list, type(contours)
    return contours

def draw_mask(image, contours, color=(0,0,0)):
    """
    >>> contours = extract_masks_one_frame("/home/drussel1/data/EIMP/EIMP_mask-RCNN_detections/Injection_Preparation.mp4.h5", 1)
    ... import cv2
    ... import numpy as np
    ... img = cv2.imread("/home/drussel1/data/EIMP/frames/000001.jpeg") 
    ... draw_mask(img, contours)
    """
    pts = [[np.asarray(c_, dtype = int) for c_ in c] for c in contours] 

    for pt in pts:
        image = cv2.fillPoly(image, pts=pt, color=(255,255,255))
    cv2.imwrite("mask.png", image)
    cv2.waitKey(0)
