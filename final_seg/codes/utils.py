import glob
import logging
import os
import re
from typing import Optional, Tuple, Union
from time import gmtime, strftime
import SimpleITK as sitk

import pickle as pkl
import torchio as tio

import cv2
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
    RandFlip
)


# ALREADY in hounsfield in pkl dump, alraedy crop, normalize, resize in npy dump, saved in numpy (float16)
def load_dicom_preprocess(
    npy_path#, 
    # center_crop_scale,
    # vol_size # torch.Size([168, 2304, 2304]), dtype=torch.float16
    # RescaleSlope=5.073579e-01,
    # RescaleIntercept=-1.000000e+03,
    # win_center: Optional[int] = 400,
    # win_width: Optional[int] = 1800,
    # x_ratio: Optional[float] = .5, 
    # y_ratio: Optional[float] = .5, 
    # z_ratio: Optional[float] = .06 # 10 / 168 slices
) -> Optional[np.ndarray]:

    img = torch.from_numpy(np.load(npy_path).astype(np.float32)) #.to(torch.float32) # torch.Size([168, 2304, 2304]), dtype=torch.float16 => [2304, 2304, 168]
    
    # time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # print(f'Done open {npy_path}... {time}')

    training_transform = Compose([
        # CenterScaleCrop(roi_scale=center_crop_scale),
        # ScaleIntensity(),
        # Resize(vol_size, mode='trilinear'),
        RandFlip(prob=.5, spatial_axis=1)
    ])

    # time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # print(f'Done processing {npy_path}... {time}')

    transformed_img = training_transform(img.unsqueeze(0))[0]       
    return transformed_img



#---------------------------- NOT USED

# max-min norm
def normalize_and_center(image): # tensor
    
    MIN_BOUND, MAX_BOUND = image.min(), image.max()
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = image - image.mean()
    
    return image


def interpolate_volume(volume: Tensor, vol_size: Optional[Tuple[int, int, int]] = None) -> Tensor:
    """Interpolate volume in last (Z) dimension

    >>> vol = torch.rand(64, 64, 12)
    >>> vol2 = interpolate_volume(vol)
    >>> vol2.shape
    torch.Size([64, 64, 64])
    >>> vol2 = interpolate_volume(vol, vol_size=(64, 64, 24))
    >>> vol2.shape
    torch.Size([64, 64, 24])
    """
    vol_shape = tuple(volume.shape)
    if not vol_size:
        d_new = min(vol_shape[:2])
        vol_size = (vol_shape[0], vol_shape[1], d_new)
    # assert vol_shape[0] == vol_shape[1], f"mixed shape: {vol_shape}"
    if vol_shape == vol_size:
        return volume
    return F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=vol_size, mode="trilinear", align_corners=False)[0, 0] # mainly for upsample, https://en.wikipedia.org/wiki/Trilinear_interpolation#:~:text=A%20geometric%20visualisation%20of%20trilinear,volume%20diagonally%20opposite%20the%20corner.




def show_volume_slice(axarr_, vol_slice, ax_name: str, v_min_max: tuple = (0., 1.)):
    axarr_[0].set_title(f"axis: {ax_name}")
    axarr_[0].imshow(vol_slice, cmap="gray", vmin=v_min_max[0], vmax=v_min_max[1])
    axarr_[1].plot(torch.sum(vol_slice, 1), list(range(vol_slice.shape[0]))[::-1])
    axarr_[1].plot(list(range(vol_slice.shape[1])), torch.sum(vol_slice, 0))
    axarr_[1].set_aspect('equal')
    axarr_[1].grid()

def idx_middle_if_none(volume: Tensor, *xyz: Optional[int]):
    xyz = list(xyz)
    vol_shape = volume.shape
    for i, d in enumerate(xyz):
        if d is None:
            xyz[i] = int(vol_shape[i] / 2)
        assert 0 <= xyz[i] < vol_shape[i]
    return xyz

def show_volume(
    volume: Tensor,
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
    fig_size: Tuple[int, int] = (14, 9),
    v_min_max: tuple = (0., 1.),
):
    """Show volume in the three axis/cuts.

    >>> show_volume(torch.rand((64, 64, 64), dtype=torch.float32))
    shape: torch.Size([64, 64, 64]), x=32, y=32, z=32  >> torch.float32
    <Figure size 1400x900 with 6 Axes>
    """
    x, y, z = idx_middle_if_none(volume, x, y, z)
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=fig_size)
    print(f"shape: {volume.shape}, x={x}, y={y}, z={y}  >> {volume.dtype}")
    show_volume_slice(axarr[:, 0], volume[x, :, :], "X", v_min_max)
    show_volume_slice(axarr[:, 1], volume[:, y, :], "Y", v_min_max)
    show_volume_slice(axarr[:, 2], volume[:, :, z], "Z", v_min_max)
    # plt.show(fig)
    return fig
