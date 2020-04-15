"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import argparse
import glob
import pdb
import os
import logging
import motmetrics as mm
import pandas as pd
import datetime
import numpy as np
from collections import OrderedDict
from pathlib import Path

from HATracking.visualize import show_tracks

OUTPUT_FORMAT_STRING = "MOT_summary_{}_{}.txt"
OUTPUT_FOLDER = "data/ADL/py_mot_metric_scores"
FRAMES_PER_CHUNK = 10000

def format_output(input_filename):
    """
    Format in the style needed for latex
    """
    with open(input_filename, "w") as infile:
        for line in infile:
            print(line.split())

def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str,
                        help='Directory containing ground truth files.')
    parser.add_argument('tests', type=str,
                        help='Directory containing tracker result files')
    parser.add_argument('--output-folder', type=str,
                        help='Where to write the score file',
                        default=OUTPUT_FOLDER)
    parser.add_argument('--loglevel', type=str, help='Log level',
                        default='info')
    parser.add_argument('--fmt', type=str, help='Data format',
                        default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    parser.add_argument('--frames-per-chunk', default=FRAMES_PER_CHUNK,
                        type=int, help='The number of frames per chunk')
    parser.add_argument('--vis', type=str, default=None,
                        help="visualize tracks. Options are 'gt', 'pred', 'both'. No input will result in no visualization")
    parser.add_argument('--video-folder', type=str,
                        help='The folder containing the ADL videos')
    return parser.parse_args()


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def ADL_scorer(args):
    gtfiles = sorted(glob.glob(os.path.join(args.groundtruths, '*')))
    tsfiles = sorted([f for f in sorted(glob.glob(os.path.join(args.tests, '*'))) if not os.path.basename(f).startswith('eval')])

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')

    print("GT files: {}\n TS files: {} ".format(gtfiles, tsfiles))
    mm.io.loadtxt(tsfiles[0], args.fmt)

    gt = OrderedDict([(Path(f).parts[-1][-8:-4], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])

    print("GT keys: {}\n TS keys: {}".format(gt.keys(),ts.keys()))

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logging.info('Running metrics')
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')
    exit()


def chunk_df(df_dict, chunk_size=10000, verbose=False):
    """
    Break the video up into chunk_size
    df_dict : OrderedDict[(string, pd.df)]
        The data
    chunk_size : int
        How many frames per chunk
    """
    output_dict = OrderedDict()
    for key, df in df_dict.items():
        index = df.index.to_frame(index=False)
        frame_ids = index['FrameId']
        max_frame = max(frame_ids)
        for i in range(0, max_frame, chunk_size):
            inds = frame_ids.between(i, i + chunk_size - 1, inclusive=True)
            # TODO determine why values is required, seems kinda dumb
            new_chunk = df.loc[inds.values, :]
            new_key = "{}_{}".format(key, i)
            output_dict[new_key] = new_chunk
            if verbose:
                print("new chunk : {}".format(new_chunk))

    return output_dict


if __name__ == '__main__':

    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    # TODO look into that movable stuff
    # sort them just for prettier output
    gtfiles = sorted(glob.glob(os.path.join(args.groundtruths, '*/gt/gt.txt')))
    tsfiles = sorted([f for f in glob.glob(os.path.join(args.tests, '*.txt'))
                     if not os.path.basename(f).startswith('eval')])

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')
    print("GT files: {}\n TS files: {} ".format(gtfiles, tsfiles))

    mm.io.loadtxt(tsfiles[0], fmt=args.fmt)

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=args.fmt,
                     min_confidence=1)) for f in gtfiles[:1]])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0],
                     mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles[:1]])
    if args.vis is not None:
        video_files = sorted(glob.glob(os.path.join(args.video_folder, "*")))
        # TODO visualize both
        first_ts = next(iter(ts.values()))  # get the first element
        first_gt = next(iter(gt.values()))
        first_video = video_files[0]
        output_file = "vis_{}.avi".format(args.vis)
        output_file = os.path.join(args.output_folder, output_file)
        
        if args.vis == "both":
            show_tracks(first_video, output_file, first_ts, first_gt)
        elif args.vis == "gt":
            show_tracks(first_video, output_file, first_gt)
        elif args.vis == "pred":
            show_tracks(first_video, output_file, first_ts)
        else:
            raise ValueError("The vis option {} was not included".format(args.vis))



    new_ts = chunk_df(ts)
    new_gt = chunk_df(gt)
    NUM_ROWS = 600000
    #ts = OrderedDict([(k, v.iloc[:NUM_ROWS, :]) for k, v in ts.items()])
    print("GT keys: {}\n TS keys: {}".format(gt.keys(), ts.keys()))
    print("new GT keys: {}\n new TS keys: {}".format(new_gt.keys(), new_ts.keys()))
    # compute the metrics

    mh = mm.metrics.create()
    accs, names = compare_dataframes(new_gt, new_ts)
    logging.info('Running metrics')

    print(mm.metrics)
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    rendered_summary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(rendered_summary)
    print(args)
    test_folder = os.path.split(args.tests)[-1]
    score_file = OUTPUT_FORMAT_STRING.format(test_folder,
                    str(datetime.datetime.now()).replace(" ", ""))
    output_file = os.path.join(args.output_folder, score_file)
    with open(output_file, "w") as outfile:
        outfile.write(rendered_summary)
        outfile.write("\n")
        outfile.write(str(args))
    logging.info('Completed')
    pdb.set_trace()
