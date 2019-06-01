import h5py
import pdb

filename = "/home/drussel1/data/ADL/detections/5_29_19/P_01.h5"

h5file = h5py.File(filename)
#print(list(h5file.keys()))
for k in h5file.keys():
    dset = h5file[k]
    print(dset.value)
pdb.set_trace()
