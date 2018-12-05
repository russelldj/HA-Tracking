import numpy as np
import cv2
import tools
#from pycocotools import mask as MaskUtil
import h5py
import json
import pdb

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
