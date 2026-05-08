"""Quick debug: print the keys and shapes inside a SEED-IV .mat file."""
import scipy.io as sio
import os

mat_path = '/fs/vulcan-projects/fsh_track/jason-bhargav-temp/CMSC472-Final/data/eeg_feature_smooth/1/1_20160518.mat'

print(f"Loading: {mat_path}\n")
data = sio.loadmat(mat_path)

for k, v in sorted(data.items()):
    if k.startswith('__'):
        continue
    if hasattr(v, 'shape'):
        print(f"  {k:30s}  shape={v.shape}  dtype={v.dtype}  ndim={v.ndim}")
    else:
        print(f"  {k:30s}  type={type(v).__name__}  value={repr(v)[:60]}")
