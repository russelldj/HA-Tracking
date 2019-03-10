import tools
import sys
import argparse

"""
THis is going to do everything
Most operations that I want to run will be functions in here which can just be called directly with or without args

"""
def computeOpenPoseKeypoints():
    sys.path.append('libs/openpose/build/python');    
    from openpose import pyopenpose as op
    print(op)

def test_H5_load():
    pass

if __name__ == "__main__":
    computeOpenPoseKeypoints()

