PYTHONPATH=/home/david/dev/HA-Tracking/HATracking/libs/SiamMask:/home/david/dev/HA-Tracking/HATracking/libs/SiamMask/experiments/siammask_sharp
#:$PYTHONPATH

conda activate siammask
alias eval_command="python eval_motchallenge.py data/ADL/annotations/MOT_style_interpolated data/ADL/outputs/experiments/first_run_no_shift/preds     --frames-per-chunk 20000 --vis gt --video-folder data/ADL/videos"
