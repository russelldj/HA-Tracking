import pandas as pd
import cv2  
import os
import pdb
import threading
from tools import tools
from libs.DaSiamRPN.code import SiamRPN_tracker# this takes a while

VIDEO_FILE = 'ADL_videos/P_{:02d}.MP4'
OUTPUT_FOLDER = 'ADL_Market_format/all_bounding_boxes'
IMG_SHAPE = (128, 128)

def track_section(track, vid, outfile):
    initial_bbox = track.iloc[0][[1,2,3,4]].tolist()
    initial_bbox = tools.ltrb_to_tlbr(initial_bbox)
    initial_bbox = tools.tlbr_to_ltwh(initial_bbox)
    initial_index = track.iloc[0]["frame"]
    obj_class = track.iloc[0]["class"]
    obj_ID = track.iloc[0]["ID"]

    final_index = track.iloc[-1]["frame"]

    vid.set(1,initial_index)
    print(initial_index)
    print(vid.get(1))
    ok, frame = vid.read()

    tracker = SiamRPN_tracker.SiamRPN_tracker(frame, initial_bbox)

    while vid.get(1) <= final_index and ok:# the current frame 
        ltwh, score, crop_region = tracker.predict(frame)
        tlbr = tools.ltwh_to_tlbr(ltwh)
        l, t, r, b = tools.tlbr_to_ltrb(tlbr)
        outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(obj_ID, l, t, r, b, int(vid.get(1)), 0, obj_class)) # TODO validate this line
        ok, frame = vid.read()
        if frame is None:
            break # this is just a messier way of doing the while check 

def run_video(start_vid, end_vid):# inclusive, exclusive
    for i in range(start_vid, end_vid):
        df = pd.read_csv('/home/drussel1/data/ADL/ADL_annotations/object_annotation/object_annot_P_{:02d}.txt'.format(i), sep=' ', 
                    names=['ID', 'x1', 'y1', 'x2', 'y2', 'frame', 'active',
                    'class', 'NaN'], index_col=None)
        vid = cv2.VideoCapture('/home/drussel1/data/ADL/ADL_videos/P_{:02d}.MP4'.format(i))
        with open("/home/drussel1/dev/TTM/TTM/outputs/P_{:02d}.txt".format(i), "w") as outfile:
            #pdb.set_trace()
            df.sort_values(by=['ID', 'frame'], inplace = True)
             
            IDs = list(set(df['ID'].tolist()))
            for ID in IDs:
                track = df.loc[df['ID'] == ID]
                track_section(track, vid, outfile)
if __name__ == "__main__": 
    # creating thread 
    pdb.set_trace()
    run_video(1,2)
    #run_video(6,11)
    #run_video(11,16)
    #run_video(16,21)
    #for i in range(3,6,2):
    #    all_threads.append(threading.Thread(target=run_video, args=(i,i+2)))

    #for a_thread in all_threads:
    #    a_thread.start()

    #for a_thread in all_threads:
    #    a_thread.join()

  
    #for index, row in df.iterrows():
    #       print(row['frame'])
    #       cap.set(1, row['frame'])
    #       ret, img = cap.read()
    #       if not ret:
    #           break
    #       crop = img[2*row['y1']:2*row['y2'],2*row['x1']:2*row['x2']].copy()
    #       crop = cv2.resize(crop, IMG_SHAPE, cv2.INTER_CUBIC)  
    #       print(crop.shape)
    #       cv2.imwrite('{}/{:02d}{:04d}_c1s1_{}_00.jpg'.format(OUTPUT_FOLDER, i, row['ID'], row['frame']), crop)
