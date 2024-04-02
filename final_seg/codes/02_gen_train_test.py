import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

fpath = '../data/slicer_annotated/latest.raw.nii'
img = nib.load(fpath)
dat = img.get_data()
fname = '../data/slicer_annotated/latest.seg.nrrd'
seg, _ = nrrd.read(fname)


# each will go random rotation (100% pct), and flip
train_seq, val_seq = train_test_split(np.array([i for i in range(115)]), shuffle=False, test_size=15) 
train_seq = resample(train_seq, n_samples=1000, random_state=24) 
val_seq = resample(val_seq, n_samples=150, random_state=24) 

for ct, idx in enumerate(train_seq):
    img_tmp = nib.Nifti1Image(dat[:,:,idx], np.eye(4))   
    seg_tmp = nib.Nifti1Image(seg[:,:,idx], np.eye(4))  

    img_tmp.to_filename(f'../data/train_x10/img{ct}_{idx}.nii.gz') 
    seg_tmp.to_filename(f'../data/train_x10/seg{ct}_{idx}.nii.gz') 

for ct, idx in enumerate(val_seq):
    img_tmp = nib.Nifti1Image(dat[:,:,idx], np.eye(4))   
    seg_tmp = nib.Nifti1Image(seg[:,:,idx], np.eye(4))  

    img_tmp.to_filename(f'../data/val_x10/img{ct}_{idx}.nii.gz') 
    seg_tmp.to_filename(f'../data/val_x10/seg{ct}_{idx}.nii.gz') 