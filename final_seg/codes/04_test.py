# to test segmentation model performance on all 2D slices for all data
# note used so far

import logging
import os
import sys
import tempfile
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
import gc

import monai
from monai.data import list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandRotated,
    RandFlipd,
    ScaleIntensityd,
)

os.chdir('/home/chenfy/projects/seg/codes')


transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]), 

        RandRotated(keys=["img", "seg"], range_x=(-90,90), prob=1),
        RandFlipd(keys=["img", "seg"], prob=.5)
    ]
)


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

val_images = sorted(glob('../data/pkl_to_2d_nii/*.nii')) #sorted(glob('../data/val/img*.nii.gz'))
val_segs = sorted(glob('../data/pkl_to_2d_nii/*.nii'))[::-1] #just use the fake data; #sorted(glob('../data/val_x10/seg*.nii.gz')) #sorted(glob('../data/val/seg*.nii.gz'))
val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_segs)]

val_ds = monai.data.Dataset(data=val_files, transform=transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 
torch.cuda.empty_cache()
gc.collect()

model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(64, 64*2, 64*4, 64*8, 64*16),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

model = nn.DataParallel(model, device_ids=[2,3])
model = model.to(device) # next(model.parameters()).is_cuda

model.load_state_dict(torch.load("../data/model_x10/best_metric_model_segmentation2d_dict_epoch.pth"))
model.eval()

with torch.no_grad():
    for val_data in val_loader:

        basename = os.path.basename(val_data['seg_meta_dict']['filename_or_obj'][0]).strip('.nii')

        val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device) ## label is fake
        # define sliding window size and batch size for windows inference
        roi_size = (512, 512) #### thu bug debugged
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
#         val_labels = decollate_batch(val_labels)
        
        
        fig, axes = plt.subplots(nrows=1, ncols=3)
        ax_list = axes.flatten()
        ax_list[0].imshow(val_images[0].cpu().detach().numpy()[0], cmap='Greys')
        ax_list[1].imshow(val_outputs[0].cpu().detach().numpy()[0], cmap='Greys')
        ax_list[2].imshow(val_images[0].cpu().detach().numpy()[0], cmap='Greys')
        ax_list[2].imshow(val_outputs[0].cpu().detach().numpy()[0], cmap='Greens', alpha=.3)
        
        fig.savefig(f'../data/nii_seg/{basename}.png')