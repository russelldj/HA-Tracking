# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from libs.SiamMask.tools.test import *
import pdb
import pandas as pd
from run_experiments import DaSiamShiftSearch, parse_args
from tools import tools, KeypointVisualization, KeypointCapture
from custom import Custom


def track_section(track, vid, args, video_writer):
    initial_bbox = track.iloc[0][[1,2,3,4]].tolist()
    initial_bbox = tools.ltrb_to_tlbr(initial_bbox)
    initial_bbox = tools.tlbr_to_ltwh(initial_bbox) # both are needed
    initial_bbox *= 2 #ADL annotations are off by two I think
    x, y, w, h = initial_bbox # initialize
    

    index = track.iloc[0]["frame"]
    obj_class = track.iloc[0]["class"]
    obj_ID = track.iloc[0]["ID"]

    final_index = track.iloc[-1]["frame"]

    vid.set(1,index)
    print(index)
    print(vid.get(1))
    ok, img = vid.read()

    # set up the tracker
    cfg = load_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    state = siamese_init(img, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker

    while index <= final_index and ok:# the current frame 
        #this is where I want to update it to grab the numpy frame
        state = siamese_track(state, img, mask_enable=True, refine_enable=True, device=device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
        cv2.polylines(img, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        cv2.imshow('SiamMask', img)
        video_writer.write(img)
        key = cv2.waitKey(1)


        ok, img = vid.read()
        if img is None:
            break # this is just a messier way of doing the while check 
        index+=1


def read_ADL_data(annotation_file):
     df = pd.read_csv(annotation_file, sep=' ',
          names=['ID', 'x1', 'y1', 'x2', 'y2', 'frame', 'active',
          'class', 'NaN'], index_col=None)
     df.sort_values(by=['ID', 'frame'], inplace = True)

     IDs = list(set(df['ID'].tolist()))
     return df, IDs

def run_video(video_file, annotation_file, args, output_video_file="visualization.avi", output_tracks=None, fps=30):
    df, IDs = read_ADL_data(annotation_file)
    video_cap = cv2.VideoCapture(video_file)
   
    ok, test_img = video_cap.read()
    height, width, _ = test_img.shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for ID in IDs:
        track = df.loc[df['ID'] == ID]
        track_section(track, video_cap, args, video_writer)
        video_writer.write(np.zeros((height, width, 3), dtype=np.uint8))
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
    
    parser.add_argument('--resume', default='', type=str, required=True,
                        metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config', default='config_davis.json',
                        help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--video', default='/home/david/data/ADL_videos/P_01.MP4', help='datasets')
    parser.add_argument('--cpu', action='store_true', help='cpu mode')
    args = parser.parse_args()
    run_video("/home/david/data/ADL_videos/P_01.MP4", "/home/david/data/ADL_annotations/object_annotation/object_annot_P_01.txt", args)
    exit()
    # creating thread 
    OUTPUT_FOLDER = "/usr0/home/drussel1/dev/TTM/TTM/data/CVPR/6_9_runs"
    #OUTPUT_FOLDER = "/usr0/home/drussel1/dev/TTM/TTM/data/CVPR/temp"
    SET_SHIFT = True
    IMSHOW = False
    OUTPUT_FILENAME = "test.avi"
    FPS = 30
    (WIDTH, HEIGHT) = (1280,960)
    args = parse_args()

    DaSiamShiftSearch().run_video(args.start,args.stop, OUTPUT_FOLDER, OUTPUT_FILENAME, WIDTH, HEIGHT)

    exit()
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Read video
    cap = cv2.VideoCapture(args.video)
    ok, img = cap.read()

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', img, False, False)
        x, y, w, h = init_rect
    except:
        exit()

 
    toc = 0
    while True:
        ok, img = cap.read()
        if not ok: break

        tic = cv2.getTickCount()
        state = siamese_track(state, img, mask_enable=True, refine_enable=True, device=device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
        cv2.polylines(img, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        #cv2.imshow('SiamMask', img)
        #key = cv2.waitKey(1)

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
