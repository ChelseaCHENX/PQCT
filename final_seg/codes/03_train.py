# unet train on segmented data
import logging
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

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
from monai.visualize import plot_2d_or_3d_image

'''
expand images (random sampling) => transforms (random rotation & flip)
'''
os.chdir('/home/chenfy/projects/seg/codes')


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    train_images = sorted(glob('../data/train_x10/img*.nii.gz')) # sorted(glob('../data/train/img*.nii.gz'))
    train_segs = sorted(glob('../data/train_x10/seg*.nii.gz')) # sorted(glob('../data/train/seg*.nii.gz'))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_segs)]
    
    val_images = sorted(glob('../data/val_x10/img*.nii.gz')) #sorted(glob('../data/val/img*.nii.gz'))
    val_segs = sorted(glob('../data/val_x10/seg*.nii.gz')) #sorted(glob('../data/val/seg*.nii.gz'))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_segs)]
       

    transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]), 

            RandRotated(keys=["img", "seg"], range_x=(-90,90), prob=1),
            RandFlipd(keys=["img", "seg"], prob=.5)
        ]
    )
    # define dataset, data loader

    train_ds = monai.data.Dataset(data=train_files, transform=transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

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
    loss_function = monai.losses.DiceLoss(sigmoid=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()

    total_epochs = 50
    for epoch in range(total_epochs):
        print("-" * total_epochs)
        print(f"epoch {epoch + 1}/{total_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                roi_size = (512, 512) # range of interest => full
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), f"../data/model/best_metric_model_segmentation2d_dict_epoch{epoch}.pth")#torch.save(model.state_dict(), "../data/model_x10/best_metric_model_segmentation2d_dict.pth")#torch.save(model.state_dict(), "../data/model/best_metric_model_segmentation2d_dict.pth")
                # torch.save(model, f'../data/model_x10/best_metric_model_segmentation2d_epoch{epoch}.pkl')#torch.save(model, f'../data/model/best_metric_model_segmentation2d_epoch{epoch}.pkl')
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()



if __name__ == "__main__":
    main()

# tmux a -t 0
# source activate pytorch
# python train.py > ../logs/121922_train.log 2>&1 #the x10 version #python unet_train.py > ../logs/120822_x10_model.log 2>&1
