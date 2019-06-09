import pandas as pd
import pdb
import os
import numpy as np

def load_ADL(fname):
    df = pd.read_csv(
                    fname,
                    sep=' ',
                    #index_col=[5,1], 
                    header=None,
                    names=["objectTrackID", "x1", "y1", "x2", "y2", "frameNumber", "active", "objectLabel"],
                    engine='python',
                    index_col=False
                    )
    # These are what we want
    return df


def ADL_to_MOT(ADL):
    output = pd.DataFrame(columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'])
    output["X"] = ADL["x1"] * 2
    output["Y"] = ADL["y1"] * 2
    output["Width"] = (ADL["x2"] - ADL["x1"]) * 2
    output["Height"] = (ADL["y2"] - ADL["y1"]) * 2
    output["FrameId"] = ADL["frameNumber"]
    output["Id"] = ADL["objectTrackID"]
    output["Confidence"] = 1
    output["ClassId"] = 1 # this could be changed
    output["Visibility"] = 1
    #output.set_index(['FrameId', 'Id'])
    print(output)
    return output

def load_MOT(fname):
    df = pd.read_csv(
                    fname,
                    sep=' ',
                    #index_col=[5,1], 
                    header=None,
                    names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'],
                    engine='python',
                    index_col=False
                    )
    return df

def remove_intermediate_frames(gt, pred):
    """
    both inputs are going to be dataframes
    """
    frames = gt["FrameId"].unique()
    indexing = pred["FrameId"].isin(frames)
    pred = pred[indexing]
    return pred


def remove_imovable_objects(gt_ADL, pred):
    """
    pass in the ground truth ADL and the MOT prediction
    find all of the objects which are in the imovable list in the gt and get their IDs
    Then remove all of the ones with this ID from the gt and the prediction 
    finally, convert the gt to MOT
    """
    IMOVABLE_TYPES = set({'bed', 'keyboard', 'tap', 'oven/stove', 'thermostat', 'person', 'blanket', 'tv', 'microwave', 'door', 'trash_can', 'fridge', 'washer/dryer', 'monitor'})

    imovable = gt_ADL["objectLabel"].isin(IMOVABLE_TYPES)
    if not np.array_equal(sorted(gt_ADL["objectTrackID"].unique()),  sorted(pred["Id"].unique())):
        #same =  np.equal(sorted(gt_ADL["objectTrackID"].unique()), sorted(pred["Id"].unique()))
        print("Detected inequality in IDs in the first stage")

    gt_ADL = gt_ADL[~imovable] # get rid of the imovable objects, note ~ for negation
    movable_IDs = gt_ADL["objectTrackID"].unique()
    pred_indices = pred["Id"].isin(movable_IDs)
    pred = pred[pred_indices]
    if not np.array_equal(sorted(gt_ADL["objectTrackID"].unique()), sorted(pred["Id"].unique())):
        print("Detected inequality in IDs in the second stage")

    gt = ADL_to_MOT(gt_ADL)
    return gt, pred


def get_all_classes():
    """
    these are {'oven/stove', 'shoe', 'kettle', 'tv', 'microwave', 'food/snack', 'person', 'towel', 'thermostat', 'vacuum', 'comb', 'tooth_paste', 'cloth', 'cell_phone', 'container', 'pills', 'bottle', 'laptop', 'elec_keys', 'mop', 'detergent', 'monitor', 'tap', 'knife/spoon/fork', 'trash_can', 'blanket', 'washer/dryer', 'keyboard', 'tv_remote', 'book', 'shoes', 'bed', 'dish', 'door', 'basket', 'electric_keys', 'milk/juice', 'tooth_brush', 'pan', 'mug/cup', 'large_container', 'cell', 'dent_floss', 'pitcher', 'perfume', 'tea_bag', 'fridge', 'soap_liquid'}

    The ones that it seems resonable to track are:
    movable = set(['kettle', 'shoe', 'food/snack', 'towel', 'vacuum', 'comb', 'tooth_paste', 'cloth', 'cell_phone', 'container', 'pills', 'bottle', 'laptop', 'elec_keys', 'mop', 'detergent', 'knife/spoon/fork',  'tv_remote', 'book', 'shoes', 'basket', 'electric_keys', 'milk/juice', 'tooth_brush', 'pan', 'mug/cup', 'large_container', 'cell', 'dent_floss', 'pitcher', 'perfume', 'tea_bag', 'dish', 'soap_liquid'])
    """
    unique_objects = set()
    for i in range(1, 21):
        GT_IN = "/home/drussel1/data/ADL/ADL_annotations/just_object_annotations/object_annot_P_{:02d}.txt".format(i)
        gt = load_ADL(GT_IN)
        new_objects = gt["objectLabel"].unique().tolist()
        unique_objects.update(new_objects)

    movable = set(['kettle', 'shoe', 'food/snack', 'towel', 'vacuum', 'comb', 'tooth_paste', 'cloth', 'cell_phone', 'container', 'pills', 'bottle', 'laptop', 'elec_keys', 'mop', 'detergent', 'knife/spoon/fork',  'tv_remote', 'book', 'shoes', 'basket', 'electric_keys', 'milk/juice', 'tooth_brush', 'pan', 'mug/cup', 'large_container', 'cell', 'dent_floss', 'pitcher', 'perfume', 'tea_bag', 'dish', 'soap_liquid'])
    not_movable = unique_objects - movable 
    print(not_movable)
    combination = movable | not_movable
    pdb.set_trace()
    assert(combination == unique_objects)
