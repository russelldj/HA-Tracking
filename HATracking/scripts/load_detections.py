import h5py
import pdb
import json

filename = "/home/drussel1/data/ADL/detections/5_29_19/P_01.h5"

h5file = h5py.File(filename)
pdb.set_trace()
#print(list(h5file.keys()))
for k in h5file.keys():
    dset = h5file[k]
    frame_data = dset.value
    frame_data = json.loads(frame_data)
    frame = frame_data["frame"]
    classes = frame_data["classes"]
    boxes = frame_data["boxes"]
    contours = frame_data["contours"]
    print(type(contours[0]))


