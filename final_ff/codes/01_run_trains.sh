#!/bin/bash

cd /home/chenfy/projects/classifier_fragilefracture/codes

# first we run fold 1 to see the general performance
/data6/cfy/anaconda3/envs/pytorch/bin/python train_1A5.py > ../logs/012824_1A5.log
/data6/cfy/anaconda3/envs/pytorch/bin/python train_1A1.py > ../logs/012824_1A1.log
/data6/cfy/anaconda3/envs/pytorch/bin/python train_1A2.py > ../logs/012824_1A2.log
/data6/cfy/anaconda3/envs/pytorch/bin/python train_1A3.py > ../logs/012824_1A3.log
/data6/cfy/anaconda3/envs/pytorch/bin/python train_1A4.py > ../logs/012824_1A4.log

# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A5.py > ../logs/fold1/012824_2A5.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A1.py > ../logs/fold1/012824_2A1.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A2.py > ../logs/fold1/012824_2A2.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A3.py > ../logs/fold1/012824_2A3.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A4.py > ../logs/fold1/012824_2A4.log

## then we run fold 2-7 for the rest
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_1A5.py > ../logs/fold2_7/012824_1A5.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_1A1.py > ../logs/fold2_7/012824_1A1.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_1A2.py > ../logs/fold2_7/012824_1A2.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_1A3.py > ../logs/fold2_7/012824_1A3.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_1A4.py > ../logs/fold2_7/012824_1A4.log

# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A5.py > ../logs/fold2_7/012824_2A5.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A1.py > ../logs/fold2_7/012824_2A1.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A2.py > ../logs/fold2_7/012824_2A2.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A3.py > ../logs/fold2_7/012824_2A3.log
# /data6/cfy/anaconda3/envs/pytorch/bin/python train_2A4.py > ../logs/fold2_7/012824_2A4.log