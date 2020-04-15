import pandas as pd
import pdb
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def load_ADL(fname):
    df = pd.read_csv(
        fname,
        sep=' ',
        # index_col=[5,1],
        header=None,
        names=[
            "objectTrackID",
            "x1",
            "y1",
            "x2",
            "y2",
            "frameNumber",
            "active",
            "objectLabel"],
        engine='python',
        index_col=False
    )
    # These are what we want
    return df


def ADL_to_MOT(ADL):
    output = pd.DataFrame(
        columns=[
            'FrameId',
            'Id',
            'X',
            'Y',
            'Width',
            'Height',
            'Confidence',
            'ClassId',
            'Visibility'])
    output["X"] = ADL["x1"] * 2
    output["Y"] = ADL["y1"] * 2
    output["Width"] = (ADL["x2"] - ADL["x1"]) * 2
    output["Height"] = (ADL["y2"] - ADL["y1"]) * 2
    output["FrameId"] = ADL["frameNumber"]
    output["Id"] = ADL["objectTrackID"]
    output["Confidence"] = 1
    output["ClassId"] = 1  # this could be changed
    output["Visibility"] = 1
    return output


def load_MOT(fname):
    df = pd.read_csv(
        fname,
        sep=' ',
        # index_col=[5,1],
        header=None,
        names=[
            'FrameId',
            'Id',
            'X',
            'Y',
            'Width',
            'Height',
            'Confidence',
            'ClassId',
            'Visibility'],
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


def run_remove_imovable():
    WRITE_OUT_GTS = True
    PRED_OUT = "/usr0/home/drussel1/dev/TTM/TTM/data/CVPR/no_imovables/not_shifted_BAD"
    if not os.path.isdir(PRED_OUT):
        os.system("mkdir -p {}".format(PRED_OUT))
    for i in range(1, 21):
        GT_IN = "/home/drussel1/data/ADL/ADL_annotations/just_object_annotations/object_annot_P_{:02d}.txt".format(
            i)
        PRED_IN = "/usr0/home/drussel1/dev/TTM/TTM/data/CVPR/scaled_wrong/not_shifted/P_{:02d}.txt".format(
            i)
        ADL_gt = load_ADL(GT_IN)
        pred = load_MOT(PRED_IN)
        # TODO filter these again to get only the ones which line up
        gt, pred = remove_imovable_objects(ADL_gt, pred)
        pred = remove_intermediate_frames(gt, pred)

        pred.to_csv(
            os.path.join(
                PRED_OUT,
                "P_{:02d}.txt".format(i)),
            sep=" ",
            header=False,
            index=False)
        if WRITE_OUT_GTS:
            GT_OUT = "/usr0/home/drussel1/dev/TTM/TTM/data/CVPR/no_imovables/gt/P_{:02d}/gt/".format(
                i)
            if not os.path.isdir(GT_OUT):
                os.system("mkdir -p {}".format(GT_OUT))
            gt.to_csv(
                os.path.join(
                    GT_OUT,
                    "gt.txt"),
                sep=" ",
                header=False,
                index=False)  # don't write a header or index column


def remove_imovable_objects(gt_ADL, pred):
    """
    pass in the ground truth ADL and the MOT prediction
    find all of the objects which are in the imovable list in the gt and get their IDs
    Then remove all of the ones with this ID from the gt and the prediction
    finally, convert the gt to MOT
    """
    IMOVABLE_TYPES = set({'bed',
                          'keyboard',
                          'tap',
                          'oven/stove',
                          'thermostat',
                          'person',
                          'blanket',
                          'tv',
                          'microwave',
                          'door',
                          'trash_can',
                          'fridge',
                          'washer/dryer',
                          'monitor'})

    imovable = gt_ADL["objectLabel"].isin(IMOVABLE_TYPES)
    if not np.array_equal(
        sorted(
            gt_ADL["objectTrackID"].unique()), sorted(
            pred["Id"].unique())):
        #same =  np.equal(sorted(gt_ADL["objectTrackID"].unique()), sorted(pred["Id"].unique()))
        print("Detected inequality in IDs in the first stage")

    # get rid of the imovable objects, note ~ for negation
    gt_ADL = gt_ADL[~imovable]
    movable_IDs = gt_ADL["objectTrackID"].unique()
    pred_indices = pred["Id"].isin(movable_IDs)
    pred = pred[pred_indices]
    if not np.array_equal(
        sorted(
            gt_ADL["objectTrackID"].unique()), sorted(
            pred["Id"].unique())):
        print("Detected inequality in IDs in the second stage")

    gt = ADL_to_MOT(gt_ADL)
    return gt, pred


def get_all_classes():
    """
    these are {'oven/stove', 'shoe', 'kettle', 'tv', 'microwave', 'food/snack',
    'person', 'towel', 'thermostat', 'vacuum', 'comb', 'tooth_paste', 'cloth', 'cell_phone', 'container', 'pills', 'bottle', 'laptop', 'elec_keys', 'mop', 'detergent', 'monitor', 'tap', 'knife/spoon/fork', 'trash_can', 'blanket', 'washer/dryer', 'keyboard', 'tv_remote', 'book', 'shoes', 'bed', 'dish', 'door', 'basket', 'electric_keys', 'milk/juice', 'tooth_brush', 'pan', 'mug/cup', 'large_container', 'cell', 'dent_floss', 'pitcher', 'perfume', 'tea_bag', 'fridge', 'soap_liquid'}

    The ones that it seems resonable to track are:
    movable = set(['kettle', 'shoe', 'food/snack', 'towel', 'vacuum', 'comb', 'tooth_paste', 'cloth', 'cell_phone', 'container', 'pills', 'bottle', 'laptop', 'elec_keys', 'mop', 'detergent', 'knife/spoon/fork',  'tv_remote', 'book', 'shoes', 'basket', 'electric_keys', 'milk/juice', 'tooth_brush', 'pan', 'mug/cup', 'large_container', 'cell', 'dent_floss', 'pitcher', 'perfume', 'tea_bag', 'dish', 'soap_liquid'])
    """
    unique_objects = set()
    for i in range(1, 21):
        GT_IN = "/home/drussel1/data/ADL/ADL_annotations/just_object_annotations/object_annot_P_{:02d}.txt".format(
            i)
        gt = load_ADL(GT_IN)
        new_objects = gt["objectLabel"].unique().tolist()
        unique_objects.update(new_objects)

    movable = set(['kettle',
                   'shoe',
                   'food/snack',
                   'towel',
                   'vacuum',
                   'comb',
                   'tooth_paste',
                   'cloth',
                   'cell_phone',
                   'container',
                   'pills',
                   'bottle',
                   'laptop',
                   'elec_keys',
                   'mop',
                   'detergent',
                   'knife/spoon/fork',
                   'tv_remote',
                   'book',
                   'shoes',
                   'basket',
                   'electric_keys',
                   'milk/juice',
                   'tooth_brush',
                   'pan',
                   'mug/cup',
                   'large_container',
                   'cell',
                   'dent_floss',
                   'pitcher',
                   'perfume',
                   'tea_bag',
                   'dish',
                   'soap_liquid'])
    not_movable = unique_objects - movable
    print(not_movable)
    combination = movable | not_movable
    pdb.set_trace()
    assert(combination == unique_objects)


def interpolate_MOT(df, method="cubic"):
    """
    fill in the blanks between frames

    df : pd.dataframe
        The MOT annotations
    method : str
        The method of interpolation. linear is the only one supported right now
    """
    interpolated_tracks = []

    track_IDs = df['Id']
    original_columns = df.columns
    for track_ID in track_IDs.unique():
        track_inds = (track_IDs == track_ID).values
        one_track = df.iloc[track_inds, :]
        interpolated_tracks.append(interpolate_track(one_track, method=method,
                                                     vis=False))
    all_tracks = pd.concat(interpolated_tracks, sort=False)
    all_tracks = all_tracks[original_columns]  # rearange the columns so it's consistent
    print("The number of rows increased by a factor of {:.2f}".format(len(all_tracks) / len(df)))
    return all_tracks


def interpolate_track(track, method="cubic", vis=True,
                      vis_chance=0.01, longest_break=30):
    """
    Interpolate a dataframe containing a single track

    track : pd.DataFrame
        A dataframe containing a single track
    method : str
        Interplation method, "linear" or "cubic"
    vis : bool
        Should you plot interpolation
    longest_break : int
        The longest gap to be filled with interpolations
    """

    # sort the track by frame ID or at least check that that's the case
    if len(track) <= 1:
        return track

    frame_Ids = track['FrameId'].values
    interpolated_dists = np.diff(frame_Ids)
    long_breaks = interpolated_dists > longest_break
    if np.any(long_breaks):
        long_break_locs = np.where(long_breaks)[0]  # TODO see if this can be made more efficient
        # I'm not entirely sure why there's an off-by-one error since this works for np
        split_tracks = np.array_split(track, long_break_locs + 1)
        interpolated_subsections = []
        for track in split_tracks:
            interpolated_track = interpolate_track(track,
                                                   method=method,
                                                   vis=vis,
                                                   vis_chance=vis_chance,
                                                   longest_break=longest_break)
            interpolated_subsections.append(interpolated_track)
        concatenated_tracks = pd.concat(interpolated_subsections, sort=False)
        return concatenated_tracks

    else:
        start = np.min(frame_Ids)
        end = np.max(frame_Ids)
        # The places we'll interpolate, all the frame values
        sampling_locations = np.arange(start, end+1)

        X1 = track['X'].values
        Y1 = track['Y'].values
        X2 = X1 + track['Width'].values
        Y2 = Y1 + track['Height'].values
        locs = np.vstack((X1, Y1, X2, Y2)).transpose()
        if method == "linear":
            f = interpolate.interp1d(frame_Ids, locs)
        elif method == "cubic":
            f = interpolate.CubicSpline(frame_Ids, locs)
        else:
            raise ValueError("Method : {} has not been implmented".format(method))

        interpolated = f(sampling_locations)
        if vis and (np.random.rand() < vis_chance):
            plt.clf()
            for i in range(4):
                plt.plot(sampling_locations, interpolated[:, i])
                plt.scatter(frame_Ids, locs[:, i])
            plt.legend(["x1", "y1", "x2", "y2"])
            plt.pause(2)

        X1 = interpolated[:, 0]
        Y1 = interpolated[:, 1]
        W = interpolated[:, 2] - X1
        H = interpolated[:, 3] - Y1
        interpolated_track = pd.DataFrame({"X": X1.astype(int),
                                           "Y": Y1.astype(int),
                                           "Width": W.astype(int),
                                           "Height": H.astype(int)})
        confidence = track.Confidence.unique()
        class_ID = track.ClassId.unique()
        visibility = track.Visibility.unique()
        Id = track.Id.unique()
        if not (len(confidence) == 1 and len(class_ID) == 1
                and len(visibility) == 1 and len(Id)):
            raise ValueError("There is variation in over the course of the track")
        interpolated_track["Confidence"] = confidence[0]
        interpolated_track["ClassId"] = class_ID[0]
        interpolated_track["Visibility"] = visibility[0]
        interpolated_track["Id"] = Id[0]
        interpolated_track["Visibility"] = visibility[0]
        interpolated_track["FrameId"] = sampling_locations
        return interpolated_track
