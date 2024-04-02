# for each 3d image, select one single 2d (512,512): mask => select tibia/humor & fibula/ulnar respectively (largest & second connected component) => find frame (left right upper lower) => project back to (2034,2034) => crop, print size => find concensus crop-size, padding to square
import os
import pickle as pkl
from glob import glob

import numpy as np
from time import gmtime, strftime
from skimage.measure import label 

import monai
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    ScaleIntensity,
    CenterScaleCrop,
    Resize,
    RandFlip
)
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

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader # not used monai dataloader, which randomized input order (making it hard to save 3d files)
import gc

os.chdir('/home/chenfy/projects/seg/codes')

# center_crop_scale = (1,1,1)
# input already cropped by (0.8,0.8,0.2) => rescaled at (512,512,10)
# skipped resize but just crop cuz want to keep resolution

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

seg_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]), 

        # RandRotated(keys=["img", "seg"], range_x=(-90,90), prob=1),
        # RandFlipd(keys=["img", "seg"], prob=.5)
    ]
)

training_transforms = Compose([
    # CenterScaleCrop(roi_scale=center_crop_scale),
    ScaleIntensity(),
    # Resize(vol_size, mode='trilinear'),
    # RandFlip(prob=.5, spatial_axis=1)
])

#---------------------------------------------- SEGMENTATION
# dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

## test run [:3]
val_images = sorted(glob('/nfs/public/fuwai_PQCT/seg_nii/*_5.nii')) #sorted(glob('../data/val/img*.nii.gz'))
val_segs = sorted(glob('/nfs/public/fuwai_PQCT/seg_nii/*_5.nii')) #just use the fake data; #sorted(glob('../data/val_x10/seg*.nii.gz')) #sorted(glob('../data/val/seg*.nii.gz'))
val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_segs)]

val_ds = monai.data.Dataset(data=val_files, transform=seg_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

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

model.load_state_dict(torch.load("/home/chenfy/projects/seg/data/model/best_metric_model_segmentation2d_dict_epoch44.pth")) # the best one
model.eval()

history_masks = {}
history_logs = {}

with torch.no_grad():
    for val_data in val_loader:

        basename = os.path.basename(val_data['seg_meta_dict']['filename_or_obj'][0]).strip('.nii') # 66_00005174_9
        # print(basename)
        pt, scan, z_idx = basename.split('_')

        if not os.path.isfile(f'{out_path}/{pt}/{scan}.npy'):
        # if True:

            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device) ## label is fake
            # define sliding window size and batch size for windows inference
            roi_size = (512, 512) #### thu bug debugged
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = post_trans(val_outputs) #[post_trans(i) for i in decollate_batch(val_outputs)] # either 0 or 1

            X = val_images[0].cpu().detach().numpy()[0] # (0,1) for all pixels, (512,512)
            mask_raw = val_outputs[0].cpu().detach().numpy()[0] # either 0 or 1 for all pixels, (512,512)
            mask = mask_raw.T # rotate to same orientation as original-2304-2304 pkl image 

            labels_1 = (getLargestCC(mask))*1 # largest connected component, eg, tibia / humor
            coords_1 = np.argwhere(labels_1==1)
            lower_1, upper_1, left_1, right_1 = min(coords_1[:,0]), max(coords_1[:,0]), min(coords_1[:,1]), max(coords_1[:,1])

            labels_2 = mask - labels_1 # smaller connected component, eg, fibula / ulna
            coords_2 = np.argwhere(labels_2==1)
            try:
                lower_2, upper_2, left_2, right_2 = min(coords_2[:,0]), max(coords_2[:,0]), min(coords_2[:,1]), max(coords_2[:,1])
            except:
                lower_2, upper_2, left_2, right_2 = -1,-1,-1,-1
            history_logs[f'{pt}_{scan}'] = [lower_1, upper_1, left_1, right_1, lower_2, upper_2, left_2, right_2]
            history_masks[f'{pt}_{scan}'] = [X, labels_1, labels_2]

            time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            print(f'Done masking {pt}_{scan} {time}')

pkl.dump(history_logs, open('../data/2d/margins.pkl', 'wb'))
pkl.dump(history_masks, open('../data/2d/masks.pkl', 'wb'))


# tmux a -t 0
# python v1_gen_seg_fromNii.py > ../logs/020223_gen_seg_fromNii2d.log 2>&1 