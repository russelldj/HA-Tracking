import matplotlib
matplotlib.use("Agg") # this turns off visualizations
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
import numpy as np
import cv2
import pdb

from . import KeypointCapture

REF_IMAGE_FNAME = "black.png"
VIDEO_OUTPUT_FNAME = "keypoint_video.avi"
REF_VIDEO_FNAME = "video1.avi"

class KeypointVisualization:
    def __init__(self, keypoint_capture):
        self.FFMpegWriter = manimation.writers['ffmpeg']
        self.metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
        self.writer = self.FFMpegWriter(fps=15, metadata=self.metadata)
        self.fig = plt.figure(figsize=(16,10))
        self.canvas = FigureCanvas(self.fig)
        self.keypoint_capture = keypoint_capture
        self.ordered_keypoints = self.keypoint_capture.GetKeypointOrdering()
        self.VIS_CONF = 0.2
        self.FINGER_THICKNESS = 8

    def TestPlot(self):
        pass
        keypoint_capture = KeypointCapture.Read2DJsonPath("video1_json", "0", "0")
        self.WritePointsToVideo(keypoint_capture)


    def WritePointsToVideo(self, keypoint_capture, video_output_file=VIDEO_OUTPUT_FNAME, reference_video=REF_VIDEO_FNAME, FRAME_RATE=30, num_frames=np.inf):
        """
        Writes the overlayed keypoints to a video 

        args
        ----------
        keypoint_capture : KeypointCapture
            The keypoints
        video_output_file : str
            Where to write the visualization
        reference_video : str
            The filename of the video the keypoints were computed from
        FRAME_RATE : int
            The output framerate of the written video
        num_frames : int
            Will write out only the first num_frames frames

        return
        ----------
        None
        """
        ordered_keys = keypoint_capture.GetKeypointOrdering()
        video_cap = cv2.VideoCapture(reference_video)

        if not video_cap.isOpened():
            print("failed to open video")
        # set up the video writer 
        num_frames = min(num_frames, keypoint_capture.num_frames)

        for i in range(num_frames):
            print("Processing frame {} of {}".format(i, num_frames))
            ret, im = video_cap.read()
            keypoint_dict = keypoint_capture.GetFrameKeypointsAsOneDict(i)
            vis_img = self.PlotSingleFrameOpenCV(keypoint_dict, ordered_keys, im)
            if i == 0:
                # create the video writer
                
                image_shape = vis_img.shape[0:2]
                vid_writer = cv2.VideoWriter(video_output_file, cv2.VideoWriter_fourcc('M','J','P','G'), FRAME_RATE, (image_shape[1], image_shape[0]))
                
            vid_writer.write(vis_img)
        vid_writer.release()
        #animation = self.camera.animate()
        #animation.save(video_output_file)
    
    def PlotSingleFrame(self, keypoint_dict, ordered_keys, im):
        plt.clf()
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        num_points = len(ordered_keys)
        color=plt.cm.rainbow(np.linspace(0,1,len(ordered_keys)))

        def getKeypointSize(i):
            # Makes the hands be plotted smaller
            return 20 if i < 25 else 2

        for i, key in enumerate(ordered_keys):
            x, y, conf = keypoint_dict[key] 
            plt.scatter(x, y, c=[color[i]], s=getKeypointSize(i))
        self.canvas.draw()
        s, (width, height) = self.canvas.print_to_buffer()
        image = np.fromstring( self.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    def PlotSingleFrameFromIndOpenCV(self, im, frame_num):
        """
        im : np.array
            The image to visualize on top of
        frame_num : int
            The frame index to acquire the keypoints with
        """
        num_points = len(self.ordered_keypoints)
        colors=plt.cm.rainbow(np.linspace(0,1,num_points))
        keypoint_dict = self.keypoint_capture.GetFrameKeypointsAsOneDict(frame_num)

        def getKeypointSize(keypoint_ind):
            # Makes the hands be plotted smaller
            return 7 if keypoint_ind < 25 else 3

        for i in range(num_points):
            key = self.ordered_keypoints[i]
            if key in keypoint_dict:
                x, y, conf = keypoint_dict[key] 
            else:
                continue # go to the next itteration because the keypoint is not in this frame
            color = [int(x * 255) for x in colors[i,0:3]]
            cv2.circle(im, (int(x), int(y)), 1 , color, getKeypointSize(i))
        return im

    def PlotSingleFrameFromAndKeypointDict(self, im, keypoint_dict, vis_conf): # TODO
        """
        im : np.array
            The image to visualize on top of
        frame_num : int
            The frame index to acquire the keypoints with
        """
        connections = np.asarray([[0,1],  [1,2],  [2,3],  [3,4],  [0,5],  [5,6],  [6,7],  [7,8],  [0,9],  [9,10],  [10,11],  [11,12],  [0,13],  [13,14],  [14,15],  [15,16],  [0,17],  [17,18],  [18,19],  [19,20]])
        colors = np.flip(np.asarray([[100,  100,  100],
        [100,    0,    0],
        [150,    0,    0],
        [200,    0,    0],
        [255,    0,    0],
        [100,  100,    0],
        [150,  150,    0],
        [200,  200,    0],
        [255,  255,    0],
        [0,  100,   50],
        [0,  150,   75],
        [0,  200,  100],
        [0,  255,  125],
        [0,   50,  100],
        [0,   75,  150],
        [0,  100,  200],
        [0,  125,  255],
        [100,    0,  100],
        [150,    0,  150],
        [200,    0,  200],
        [255,    0,  255]], dtype=np.uint8), axis=1)
        num_points = len(self.ordered_keypoints)


        #left_keypoints  = dict([(key[(1+len("left")):], value) for key, value in keypoint_dict.items() if 'left' in key])
        #right_keypoints = dict([(key[(1+len("right")):], value) for key, value in keypoint_dict.items() if 'right' in key])
        #if not len(left_keypoints) + len(right_keypoints) == len(keypoint_dict):
        #    pdb.set_trace()

        def getKeypointSize(keypoint_ind):
            # Makes the hands be plotted smaller
            return 7 if keypoint_ind < 25 else 3

        #Plot the hand points
        for i in range(num_points):
            key = self.ordered_keypoints[i]
            if key in keypoint_dict:
                x, y, conf = keypoint_dict[key]  # TODO threshold on conf
                if conf < vis_conf: # don't visualize if it's not high confidence
                    continue
            else:
                continue # go to the next itteration because the keypoint is not in this frame
            color_ind = (i - 25) % 21
            #print(color_ind)
            color = colors[color_ind, :]
            cv2.circle(im, (int(x), int(y)), 1 , color.tolist(), 15)

        # Plot the connections, i. e. fingers

        for connection in connections:
            for side in [0, 1]:
                offset = side * 21 + 25 # shift by the correct number of points in the array
                key1 = self.ordered_keypoints[connection[0] + offset]
                key2 = self.ordered_keypoints[connection[1] + offset]
                if key1 in keypoint_dict and key2 in keypoint_dict:
                    x1, y1, conf1 = keypoint_dict[key1] 
                    x2, y2, conf2 = keypoint_dict[key2] 
                    if conf1 < self.VIS_CONF or conf2 < self.VIS_CONF: # don't visualize if it's not high confidence
                        continue
                else:
                    continue # go to the next itteration because the keypoint is not in this frame
                color = colors[connection[0],:].tolist() # get the color of the first point
                cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), color, self.FINGER_THICKNESS) # repeat the last color

        return im


    def PlotSingleFrameFromIndOpenCVOpenPose(self, im, frame_num):
        """
        im : np.array
            The image to visualize on top of
        frame_num : int
            The frame index to acquire the keypoints with
        """
        connections = np.asarray([[0,1],  [1,2],  [2,3],  [3,4],  [0,5],  [5,6],  [6,7],  [7,8],  [0,9],  [9,10],  [10,11],  [11,12],  [0,13],  [13,14],  [14,15],  [15,16],  [0,17],  [17,18],  [18,19],  [19,20]])
        colors = np.flip(np.asarray([[100,  100,  100],
        [100,    0,    0],
        [150,    0,    0],
        [200,    0,    0],
        [255,    0,    0],
        [100,  100,    0],
        [150,  150,    0],
        [200,  200,    0],
        [255,  255,    0],
        [0,  100,   50],
        [0,  150,   75],
        [0,  200,  100],
        [0,  255,  125],
        [0,   50,  100],
        [0,   75,  150],
        [0,  100,  200],
        [0,  125,  255],
        [100,    0,  100],
        [150,    0,  150],
        [200,    0,  200],
        [255,    0,  255]], dtype=np.uint8), axis=1)
        num_points = len(self.ordered_keypoints)
        #colors=plt.cm.rainbow(np.linspace(0,1,num_points))
        keypoint_dict = self.keypoint_capture.GetFrameKeypointsAsOneDict(frame_num)

        #pdb.set_trace()
        def getKeypointSize(keypoint_ind):
            # Makes the hands be plotted smaller
            return 7 if keypoint_ind < 25 else 3

        #Plot the hand points
        for i in range(num_points):
            key = self.ordered_keypoints[i]
            if key in keypoint_dict:
                x, y, conf = keypoint_dict[key]  # TODO threshold on conf
                if conf < self.VIS_CONF: # don't visualize if it's not high confidence
                    continue
            else:
                continue # go to the next itteration because the keypoint is not in this frame
            color_ind = (i - 25) % 21
            #print(color_ind)
            color = colors[color_ind, :]
            cv2.circle(im, (int(x), int(y)), 1 , color.tolist(), 15)

        # Plot the connections, i. e. fingers

        for connection in connections:
            for side in [0, 1]:
                offset = side * 21 + 25 # shift by the correct number of points in the array
                key1 = self.ordered_keypoints[connection[0] + offset]
                key2 = self.ordered_keypoints[connection[1] + offset]
                if key1 in keypoint_dict and key2 in keypoint_dict:
                    x1, y1, conf1 = keypoint_dict[key1] 
                    x2, y2, conf2 = keypoint_dict[key2] 
                    if conf1 < self.VIS_CONF or conf2 < self.VIS_CONF: # don't visualize if it's not high confidence
                        continue
                else:
                    continue # go to the next itteration because the keypoint is not in this frame
                color = colors[connection[0],:].tolist() # get the color of the first point
                cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), color, self.FINGER_THICKNESS) # repeat the last color

        return im


    def PlotSingleFrameOpenCV(self, keypoint_dict, ordered_keys, im):
        num_points = len(ordered_keys)
        colors=plt.cm.rainbow(np.linspace(0,1,len(ordered_keys)))

        def getKeypointSize(i):
            # Makes the hands be plotted smaller
            return 7 if i < 25 else 3

        for i, key in enumerate(ordered_keys):
            x, y, conf = keypoint_dict[key] 
            color = [int(x * 255) for x in colors[i,0:3]]
            cv2.circle(im, (int(x), int(y)), 1 , color, getKeypointSize(i))
        return im
    
    
if __name__ == "__main__":
    #KeypointVisualization().TestWriteToVideo()
    KeypointVisualization().TestPlot()
