import os

import cv2
import numpy as np
import pdb

from HATracking.tools.TrackFiles import load_ADL, ADL_to_MOT, load_MOT
from HATracking.tools.KeypointVisualization import KeypointVisualization as kv
from HATracking.tools.KeypointCapture import KeypointCapture as cp

np.random.seed(0)  # keep it uniform between runs
VIS_COLORS = np.random.randint(0, 256, (1000, 3))


def convert_numpy(numpy_data):
    ORDERED_KEYPOINTS_HAND = [
        "Wrist",
        "Thumb1",
        "Thumb2",
        "Thumb3",
        "Thumb4",
        "Index1",
        "Index2",
        "Index3",
        "Index4",
        "Middle1",
        "Middle2",
        "Middle3",
        "Middle4",
        "Ring1",
        "Ring2",
        "Ring3",
        "Ring4",
        "Pinky1",
        "Pinky2",
        "Pinky3",
        "Pinky4"
    ]
    HANDS = ["Left", "Right"]
    assert(numpy_data.shape == (2, 1, 21, 3))
    output = dict()
    for which_hand, hand in enumerate(numpy_data):
        for which_finger, finger in enumerate(hand[0]):  # weirdextradimension
            ID = "{}_{}".format(
                HANDS[which_hand],
                ORDERED_KEYPOINTS_HAND[which_finger])
            output[ID] = finger
    return output


def show_openpose():
    VIDEO = "/home/drussel1/data/ADL/ADL_videos/P_01.MP4"
    OPENPOSE_FOLDER = "/home/drussel1/data/ADL/openpose_keypoints/keypoint_01"
    OUTPUT_FILENAME = "/home/drussel1/dev/TTM/TTM/data/CVPR/visualizations/openpose_at_{}_conf.avi"
    FPS = 30
    visualizer = kv(cp())

    for vis_threshold in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
        frame_ind = 0
        cap = cv2.VideoCapture(VIDEO)
        ok, img = cap.read()

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(OUTPUT_FILENAME.format(
            vis_threshold), fourcc, FPS, (img.shape[1], img.shape[0]))

        cap.set(1, 0)  # set it back to the first frame

        while True:
            if frame_ind % 1000 == 0:
                print(
                    "frame num {} with a confidences of {}".format(
                        frame_ind, vis_threshold))
            ok, img = cap.read()
            if not ok:
                break

            openpose_file = os.path.join(
                OPENPOSE_FOLDER, "keypoints_{:09d}.npy".format(frame_ind))
            keypoints_npy = np.load(openpose_file)
            keypoints_dict = convert_numpy(keypoints_npy)

            img = visualizer.PlotSingleFrameFromAndKeypointDict(
                img, keypoints_dict, vis_threshold)
            #cv2.imshow("plain", img)
            video_writer.write(img)
            # cv2.waitKey(100)

            frame_ind += 1
        video_writer.release()
        cap.release()


def vis_one_track_frame(frame, tracks, is_gt=False, linewidth=5, show_ID=True):
    """
    image and dataframe

    is_gt : bool
        gt annotations are marked with a black line
    linewidth : int
        The thickness of the line
    show_ID : bool
        Draw the numeric ID
    """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1

    for index, track in tracks.iterrows():
        x1 = (track["X"])
        y1 = (track["Y"])
        x2 = (track["X"] + track["Width"])
        y2 = (track["Y"] + track["Height"])
        tl = (x1, y1)
        tr = (x1, y2)
        br = (x2, y2)
        ID = track.name[1]  # The name is the (frame, ID) tuple
        color = VIS_COLORS[ID % 1000, :].tolist()
        cv2.rectangle(frame, tl, br, tuple(color), linewidth)
        if is_gt:
            cv2.rectangle(frame, tl, br, BLACK, int(linewidth / 2))
            cv2.putText(frame, str(ID), tr, FONT, FONT_SCALE, WHITE)
        else:
            cv2.putText(frame, str(ID), tl, FONT, FONT_SCALE, WHITE)  # put it on different sides

    return frame


def show_tracks(input_video_fname, output_video_fname, pred_df, gt_df=None,
                output_FPS=15):
    """
    input_video_fname : str
        The file to read from
    output_video_fname : str
        the file to write to
    pred_df : pd.DataFrame
        predictions read in in the MOT format
    gt_df : pd.DataFrame
        ground truths read in in the MOT format
    output_FPS : int
        The output framerate

    Annoyingly
    """
    #visualizer = kv(cp())

    frame_ind = 0
    cap = cv2.VideoCapture(input_video_fname)
    ok, img = cap.read()

    if not ok:
        raise ValueError("The video at {} did not open successfully".format(input_video_fname))

    # set up the output writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(
        output_video_fname, fourcc, output_FPS, (img.shape[1], img.shape[0]))

    # sort the data for slightly better performace
    # Also extract the index and massage it from the weird multiindex
    new_pred_index = pred_df.index.to_frame(index=False)
    new_pred_index.sort_values("FrameId")
    pred_df.sort_values("FrameId")
    if gt_df is not None:
        new_gt_index = gt_df.index.to_frame(index=False)
        new_gt_index.sort_values("FrameId")
        gt_df.sort_values("FrameId")

    while True:
        if frame_ind % 1000 == 0:
            print("frame num {}".format(frame_ind))

        current_locs = (new_pred_index["FrameId"] == frame_ind).values
        current_preds = pred_df.iloc[current_locs, :]
        img = vis_one_track_frame(img, current_preds)
        if gt_df is not None:
            current_locs = (new_gt_index["FrameId"] == frame_ind).values
            current_gts = gt_df.iloc[current_locs, :]
            img = vis_one_track_frame(img, current_gts, is_gt=True)

        video_writer.write(img)

        frame_ind += 1
        ok, img = cap.read()
        if not ok:
            break

    video_writer.release()
    cap.release()


def visualize_folders(video_folder, prediction_folder, groundtruth_folder):
    pass


if __name__ == "__main__":
    show_tracks()
