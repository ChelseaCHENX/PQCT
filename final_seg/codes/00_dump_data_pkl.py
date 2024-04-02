# convert dicom (from scanner) and save as pkl
# copied from archive/pqct/v0_codes/dump_data_pkl.py

from genericpath import isfile
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
import pickle
import pydicom
from time import gmtime, strftime


def check_data(path, num_files=167):
    '''
    check if all subfolders within the main folder (id) contain *.DCM files (should be 167 files)
    '''
    if os.path.isdir(path):

        dcm_len = []
        for dir in os.listdir(path):
            if os.path.isdir(f'{path}/{dir}'):
                dcm = [x for x in glob.glob(f'{path}/{dir}/*.DCM', recursive=True)]
                dcm_len.append(len(dcm))
        if np.any(np.array(dcm_len) >= 50): # any subfolder with >= 50 dcm files
            return True
        else:
            return False

src_paths = ['/nfs/public/fuwai_PQCT/images1', '/nfs/public/fuwai_PQCT/images2']
out_path = '/nfs/public/fuwai_PQCT/img'

for src_path in src_paths:
    
    for fname in os.listdir(src_path):
        
        fpath = os.path.join(src_path, fname)
        
        if check_data(fpath):
            if not os.path.isdir(f'{out_path}/{fname}'):
                os.system(f'mkdir {out_path}/{fname}')

            for subfpath in os.listdir(fpath):
                if not os.path.isdir(f'{out_path}/{fname}'):
                    os.system(f'mkdir {out_path}/{fname}/{subfpath}')

                out = f'{out_path}/{fname}/{subfpath}'

                if not os.path.isfile(f'{out}.pkl'):

                    dcm_files = [p for p in glob.glob(f'{fpath}/{subfpath}/*.DCM', recursive=True)]
                    pixels_3d = []

                    if len(dcm_files) > 50:
                        for file in dcm_files:                        
                            try:
                                dicom = pydicom.dcmread(file)
                                RescaleIntercept = dicom.RescaleIntercept
                                RescaleSlope = dicom.RescaleSlope
                                pixels_2d = dicom.pixel_array#, dicom.values()
                                pixels_2d_hu = pixels_2d * RescaleSlope + RescaleIntercept
                                pixels_3d.append(pixels_2d)
                            except:
                                print(f'Error in reading {file}')
                    pixels_3d = np.array(pixels_3d)

                    if np.all(np.array(pixels_3d.shape) > 1):
                        
                        pixels_3d = torch.tensor(pixels_3d, dtype=torch.float16)
                    
                        with open(f'{out}.pkl', 'wb') as handle:
                            pickle.dump(pixels_3d, handle)
                            time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                            print(f'{time} -- Dumping data of {pixels_3d.shape} in {out}')
                    else:
                        print( f'{src_path}/{fname}/{subfpath} is not 3d array')
                else:
                    print(f'{out}.pkl already exists')
        else:
            print(f'Incomplete files in {fpath}')

# cd /nfs/public/fuwai_PQCT/img
# ls | wc -l