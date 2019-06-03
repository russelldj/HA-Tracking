import pandas as pd
#import cv2  
import pdb

VIDEO_FILE = 'ADL_videos/P_{:02d}.MP4'
OUTPUT_FOLDER = 'ADL_Market_format/all_bounding_boxes'
IMG_SHAPE = (128, 128)

for i in range(13,21):
    df = pd.read_csv('/home/drussel1/data/ADL/ADL_annotations/object_annotation/object_annot_P_{:02d}.txt'.format(i), sep=' ', 
                names=['ID', 'x1', 'y1', 'x2', 'y2', 'frame', 'active',
                'class', 'NaN'], index_col=None)
    pdb.set_trace()
    df.sort_values(by=['ID', 'frame'], inplace = True)
    
    print(df)
    print(VIDEO_FILE.format(i))
    cap = cv2.VideoCapture(VIDEO_FILE.format(i))
    print(cap)
    
    for index, row in df.iterrows():
           print(row['frame'])
           cap.set(1, row['frame'])
           ret, img = cap.read()
           if not ret:
               break
           crop = img[2*row['y1']:2*row['y2'],2*row['x1']:2*row['x2']].copy()
           crop = cv2.resize(crop, IMG_SHAPE, cv2.INTER_CUBIC)  
           print(crop.shape)
           cv2.imwrite('{}/{:02d}{:04d}_c1s1_{}_00.jpg'.format(OUTPUT_FOLDER, i, row['ID'], row['frame']), crop)
