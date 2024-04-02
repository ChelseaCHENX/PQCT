import glob
import logging
import os
import sys
import re
import json
from typing import Optional, Tuple, Union
from time import gmtime, strftime
# import SimpleITK as sitk
from collections import Counter

import pickle as pkl
# import torchio as tio
from sklearn.model_selection import KFold

# import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers import apply_voi_lut
from torch import Tensor

import seaborn as sns

from monai.transforms import (
    Compose,
    ScaleIntensity,
    CenterScaleCrop,
    Resize,
    RandFlip,
    RandRotate
)

def _tuple_int(t: Tensor) -> tuple:
    return tuple(t.numpy().astype(int))

def resize_volume(volume: Tensor, size: int = 128) -> Tensor:
    """Resize volume with preservimg aspect ration and being centered

    >>> vol = torch.rand(64, 64, 48)
    >>> vol = resize_volume(vol, 32)
    >>> vol.shape
    torch.Size([32, 32, 32])
    """
    shape_old = torch.tensor(volume.shape)
    shape_new = torch.tensor([size] * 3)
    scale = torch.max(shape_old.to(float) / shape_new)
    shape_scale = shape_old / scale
    # print(f"{shape_old} >> {shape_scale} >> {shape_new}")
    vol_ = F.interpolate(
        volume.unsqueeze(0).unsqueeze(0), size=_tuple_int(shape_scale), mode="trilinear", align_corners=False
    )[0, 0]
    offset = _tuple_int((shape_new - shape_scale) / 2)
    volume = torch.zeros(*_tuple_int(shape_new), dtype=volume.dtype)
    shape_scale = _tuple_int(shape_scale)
    volume[offset[0]:offset[0] + shape_scale[0], offset[1]:offset[1] + shape_scale[1],
           offset[2]:offset[2] + shape_scale[2]] = vol_ #ie, the padding
    return volume

# ALREADY in hounsfield in pkl dump, alraedy crop, normalize, resize in npy dump, saved in numpy (float16)
def load(
    npy_path, 
) -> Optional[np.ndarray]:

    img = torch.from_numpy(np.load(npy_path).astype(np.float32)) #.to(torch.float32) # torch.Size([168, 2304, 2304]), dtype=torch.float16 => [2304, 2304, 168]

    return img

def preprocess(
    img, 
    mode='dev',
) -> Optional[np.ndarray]:

    dev_transform = Compose([
        RandRotate(range_x=0, range_y=0, range_z=90, prob=1), # for train_v2.py, change prob to 1 after augmentation
        RandFlip(prob=.5, spatial_axis=0) # left / right flip
    ])

    if mode=='dev':
        transformed_img = dev_transform(img.unsqueeze(0))[0] #   (256,256,10)

    elif mode=='test':
        transformed_img = img

    return transformed_img

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data6/cfy/anaconda3/lib
# export LD_LIBRARY_PATH=/data6/cfy/anaconda3/lib:/data6/cfy/anaconda3/envs/pytorch/lib
# ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found

from sklearn.utils import resample
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score
from torchmetrics import AUROC

from pandas.testing import assert_frame_equal
from sklearn import decomposition
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn.manifold import TSNE