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
from collections import OrderedDict
from pathlib import Path

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

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')   
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
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

if __name__ == '__main__':

    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    for extension in ["filtered_movable_active", "filtered_movable", "filtered"]:# run in order of least to greatest running time
        ground_truth_dir = os.path.join(args.groundtruths, extension)
        test_dir = os.path.join(args.tests, extension)


        gtfiles = sorted(glob.glob(os.path.join(ground_truth_dir, '*/gt/gt.txt'))) # sort them just for prettier output
        tsfiles = sorted([f for f in glob.glob(os.path.join(test_dir, '*.txt')) if not os.path.basename(f).startswith('eval')])

        logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
        logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
        logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
        logging.info('Loading files.')

        print("GT files: {}\n TS files: {} ".format(gtfiles, tsfiles))
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])    

        print("GT keys: {}\n TS keys: {}".format(gt.keys(),ts.keys()))

        mh = mm.metrics.create()    
        accs, names = compare_dataframes(gt, ts)
        logging.info('Running metrics')
        
        summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
        rendered_summary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        print(rendered_summary)
        print(args)
        test_folder = os.path.split(test_dir)[-1]
        with open("outputs/MOT_summary_{}_{}.txt".format(test_folder, str(datetime.datetime.now()).replace(" ", "")),"w") as outfile:
            outfile.write(rendered_summary)
            outfile.write("\n")
            outfile.write(str(args))
            outfile.write(extension)
        logging.info('Completed')
    pdb.set_trace()
