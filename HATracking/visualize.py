import cv2
import os
import glob
import numpy as np
import pdb
import inspect

from tools.TrackFiles import load_ADL, ADL_to_MOT, load_MOT
from tools.KeypointVisualization import KeypointVisualization as kv
from tools.KeypointCapture import KeypointCapture as cp

np.random.seed(0) # keep it uniform between runs
VIS_COLORS = np.random.randint(0, 256, (1000, 3))

def convert_numpy(numpy_data):
    ORDERED_KEYPOINTS_HAND =[
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
        for which_finger,finger in enumerate(hand[0]):#weirdextradimension
            ID="{}_{}".format(HANDS[which_hand],ORDERED_KEYPOINTS_HAND[which_finger])
            output[ID]=finger
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
        video_writer = cv2.VideoWriter(OUTPUT_FILENAME.format(vis_threshold), fourcc, FPS, (img.shape[1], img.shape[0]))

        cap.set(1, 0) # set it back to the first frame

        while True:
            if frame_ind % 1000 == 0:
                print("frame num {} with a confidences of {}".format(frame_ind, vis_threshold))
            ok, img = cap.read()
            if not ok:
                break

            openpose_file = os.path.join(OPENPOSE_FOLDER, "keypoints_{:09d}.npy".format(frame_ind))
            keypoints_npy = np.load(openpose_file)
            keypoints_dict = convert_numpy(keypoints_npy)

            img = visualizer.PlotSingleFrameFromAndKeypointDict(img, keypoints_dict, vis_threshold)
            #cv2.imshow("plain", img)
            video_writer.write(img)
            #cv2.waitKey(100)

            frame_ind+=1
        video_writer.release()
        cap.release()


def vis_one_track_frame(frame, tracks):
    """
    image and dataframe 
    """
    for index, track in tracks.iterrows():
        x1 = (track["X"])
        y1 = (track["Y"])
        x2 = (track["X"] + track["Width"])
        y2 = (track["Y"] + track["Height"])
        ID = track["Id"]
        color = VIS_COLORS[ID%1000,:].tolist()
        #color = [np.asscalar(c) for c in color]
        cv2.rectangle(frame,(x1, y1),(x2, y2),tuple(color),3)
    return frame

def show_tracks():
    # take a track output and a video and put the tracks onto the video
    VIDEO = "/home/drussel1/data/ADL/ADL_videos/P_01.MP4" 
    TRACK_FILE = "/home/drussel1/dev/TTM/TTM/data/CVPR/scaled_right/shifted_0.2/P_01.txt"
    OUTPUT_FILENAME = "/home/drussel1/dev/TTM/TTM/data/CVPR/visualizations/shifted_0.2_tracks.avi"
    FPS = 15
    #visualizer = kv(cp()) 

    frame_ind = 0
    cap = cv2.VideoCapture(VIDEO)
    ok, img = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (img.shape[1], img.shape[0]))

    cap.set(1, frame_ind) # set it back to the first frame

    #read the tracks in as a dataframe
    tracks = load_MOT(TRACK_FILE)
    tracks.sort_values("FrameId")



    while True:
        if frame_ind % 1000 == 0:
            print("frame num {}".format(frame_ind))
        ok, img = cap.read()
        if not ok:
            break

        #print(tracks[tracks["FrameId"] == frame_ind])
        img = vis_one_track_frame(img, tracks[tracks["FrameId"] == frame_ind])
        #cv2.imshow("plain", img)
        video_writer.write(img)
        #cv2.waitKey(100)

        frame_ind+=1
    video_writer.release()
    cap.release()

if __name__ == "__main__":
    show_tracks()
