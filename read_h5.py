import h5py
import matplotlib.pylab as plb

with h5py.File('test_data.h5', 'r') as hf:
    plb.imshow(hf["test_set_x"][100])
    plb.show()
    print(hf['list_classes'][()])



