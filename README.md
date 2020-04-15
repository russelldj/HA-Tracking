# Overview
This repository is used for tracking hand-manipulated objects

# This will be cleaned up in the very near future, please open an issue if you have a question and I will do my best to respond.
Make sure that when you clone you do `git clone --recursive` as there are included submodules

# Quickstart
I am

Download the data from: https://www.csee.umbc.edu/~hpirsiav/papers/ADLdataset/

# Useful
`HATracking/parse_ADL.py` convert from the ADL format to MOT15

# Dependancies
 pip install pymotmetrics
 pip install git+https://github.com/nwojke/pymotutils

# Data
The data is structured as follows
data\
  ADL\
    annotations\ # any sort of annotations, even generated
      action_annotations\ # the downloaded action annotations from ADL
      object_annotation\ # The downloaded object annotations from ADL, currently unused
      MOT_style_{}\ # The annotations formated in the MOT style, possibly interpolated
    outputs\ # any sort of output
      experiments\ # One folder per experiment
        ex_n\ # the folder for one experiment
          notes.md # any notes on the experiment
          scores.txt # or something like that, the scores
          preds\ # the predicted trajectories
          vis\ # The visualizations of the paths
      mediapipe\ # random mediapipe outputs
      archieved\ # stuff I don't want to deal with
    videos\ # raw videos
      P_{d:02}.mp4 # The raw ADL videos
