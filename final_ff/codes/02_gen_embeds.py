import sys
import monai
import os
root_dir = '/home/chenfy/projects/classifier_fragilefracture'
os.chdir(f'{root_dir}/codes')

from utils import *

from sklearn.model_selection import KFold
from sklearn.utils import resample

import torch
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate,
    RandFlip,
    ScaleIntensity,
    SpatialPad,
    CenterSpatialCrop
)
from monai.data import decollate_batch, DataLoader
from monai.networks.nets import resnet34
from monai.metrics import ROCAUCMetric

trial_name = sys.argv[1] #'1C2' # eg, 1C2
dim = 128
class_out = 2

class CustomResNet2_4channel(torch.nn.Module):
    def __init__(self, num_classes, class_out):
        super().__init__()
        net = resnet34(spatial_dims=2, n_input_channels=4, num_classes=num_classes)

        self.conv = torch.nn.Sequential(*list(net.children())[:-1])
        self.fc = net.fc

        self.classifier = torch.nn.Linear(num_classes,class_out) # MBD types, here is 2 || CrossEntropyLoss in PyTorch is already implemented with Softmax
        self.entropy = torch.nn.CrossEntropyLoss()

        self.sex = torch.nn.Linear(num_classes,2)
        self.entropy_bin = torch.nn.CrossEntropyLoss()

        self.agewtht = torch.nn.Linear(num_classes,3) # age, wt, ht
        self.mse_agewtht = torch.nn.MSELoss()

        self.pqcts = torch.nn.Linear(num_classes,26) # 26 pqct params for single image
        self.mse_pqct = torch.nn.MSELoss()

        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        
        # x = self.dropout(x) 
        x = self.conv(x) 
        x = x.view(-1, 512) # 512 is instrinsic from 2nd-last layer
        x = self.fc(x) # embeds - drop activation

        return x
    
class CustomResNet2_singlechannel(torch.nn.Module):
    def __init__(self, num_classes, class_out):
        super().__init__()
        net = resnet34(spatial_dims=2, n_input_channels=1, num_classes=num_classes)

        self.conv = torch.nn.Sequential(*list(net.children())[:-1])
        self.fc = net.fc
        
        self.classifier = torch.nn.Linear(num_classes,class_out) # MBD types, here is 2 || CrossEntropyLoss in PyTorch is already implemented with Softmax
        self.entropy = torch.nn.CrossEntropyLoss()

        self.sex = torch.nn.Linear(num_classes,2)
        self.entropy_bin = torch.nn.CrossEntropyLoss()

        self.agewtht = torch.nn.Linear(num_classes,3) # age, wt, ht
        self.mse_agewtht = torch.nn.MSELoss()

        self.pqcts = torch.nn.Linear(num_classes,26) # 26 pqct params for single image
        self.mse_pqct = torch.nn.MSELoss()

        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        
        # x = self.dropout(x) 
        x = self.conv(x) 
        x = x.view(-1, 512) # 512 is instrinsic from 2nd-last layer
        x = self.activation(self.fc(x)) # embeds

        return x

# define transforms
transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity(), SpatialPad(spatial_size=(768,768)), CenterSpatialCrop(roi_size=(768,768))])

# define dataset
class MbdDataset_4channel(torch.utils.data.Dataset):
    def __init__(self, image_files1, image_files2, image_files3, image_files4,   labels, transforms, params):
        self.image_files1 = image_files1 # tibia_path
        self.image_files2 = image_files2 # fibula_path
        self.image_files3 = image_files3 # radius_path
        self.image_files4 = image_files4 # ulna_path        
        self.labels = labels # n
        self.params = params # (n,29) 3+26=29: age, gender, bonetype, 'RADIUS_Tt_Ar','RADIUS_Tt_vBMD','RADIUS_Tb_Ar','RADIUS_Tb_vBMD','RADIUS_BV/TV','RADIUS_Tb_N','RADIUS_Tb_Th','RADIUS_Tb_Sp','RADIUS_Ct_Ar','RADIUS_Ct_vBMD','RADIUS_Ct_Pm','RADIUS_Ct_Po','RADIUS_Ct_Th','TIBIA_Tt_Ar','TIBIA_Tt_vBMD','TIBIA_Tb_Ar','TIBIA_Tb_vBMD','TIBIA_BV/TV','TIBIA_Tb_N','TIBIA_Tb_Th','TIBIA_Tb_Sp','TIBIA_Ct_Ar','TIBIA_Ct_vBMD','TIBIA_Ct_Pm','TIBIA_Ct_Po','TIBIA_Ct_Th'
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transforms(self.image_files1[index]), self.transforms(self.image_files2[index]), self.transforms(self.image_files3[index]), self.transforms(self.image_files4[index]), self.labels[index], self.params[index,:]

class MbdDataset_singlechannel(torch.utils.data.Dataset):
    def __init__(self, mode, image_files1, image_files2, image_files3, image_files4,   labels, transforms, params):
        self.mode = mode - 1
        self.image_files1 = image_files1 # tibia_path
        self.image_files2 = image_files2 # fibula_path
        self.image_files3 = image_files3 # radius_path
        self.image_files4 = image_files4 # ulna_path  
        self.image_files = [image_files1,image_files2,image_files3,image_files4]     
        self.labels = labels # n
        self.params = params # (n,29) 3+26=29: age, gender, bonetype, 'RADIUS_Tt_Ar','RADIUS_Tt_vBMD','RADIUS_Tb_Ar','RADIUS_Tb_vBMD','RADIUS_BV/TV','RADIUS_Tb_N','RADIUS_Tb_Th','RADIUS_Tb_Sp','RADIUS_Ct_Ar','RADIUS_Ct_vBMD','RADIUS_Ct_Pm','RADIUS_Ct_Po','RADIUS_Ct_Th','TIBIA_Tt_Ar','TIBIA_Tt_vBMD','TIBIA_Tb_Ar','TIBIA_Tb_vBMD','TIBIA_BV/TV','TIBIA_Tb_N','TIBIA_Tb_Th','TIBIA_Tb_Sp','TIBIA_Ct_Ar','TIBIA_Ct_vBMD','TIBIA_Ct_Pm','TIBIA_Ct_Po','TIBIA_Ct_Th'
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transforms(self.image_files[self.mode][index]), self.labels[index], self.params[index,:] #, self.transforms(self.image_files2[index]), self.transforms(self.image_files3[index]), self.transforms(self.image_files4[index]), self.labels[index], self.params[index,:]

# meta = pd.read_csv('../data/meta/data_index_super_sel.csv') # pt_scan, 0-5
meta = pd.read_csv('../data/meta/chivos_fragfrax_final.csv') # pt_scan, 0-5

image_files_list1, image_files_list2, image_files_list3, image_files_list4 = meta['tibia_path'], meta['fibula_path'], meta['radius_path'], meta['ulna_path']
image_params = meta[['ptid']]
image_label = meta['fragility_fracture']

bs = 5

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # has to be the first device on list
print(device)


for fold in range(1,7):

    if trial_name in ['1A1','1A2','1A3','1A4', '2A1','2A2','2A3','2A4']:
        model = CustomResNet2_singlechannel(dim, class_out).to(device)
    elif trial_name in ['1A5','2A5']:
        model = CustomResNet2_4channel(dim, class_out).to(device)
    model.load_state_dict(torch.load(f"../data/model/{trial_name}/fold{fold}.pth", map_location=device), strict=False)

    model.eval()
    with torch.no_grad():

        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        _params_list = torch.tensor([], dtype=torch.long, device=device)

        all_x1, all_x2, all_x3, all_x4 = image_files_list1, image_files_list2, image_files_list3, image_files_list4
        all_y = image_label
        all_params = image_params.values 

        if trial_name in ['1A1','1A2','1A3','1A4', '2A1','2A2','2A3','2A4']:
            mode = 1 if trial_name in ['1A1','2A1'] else 2 if trial_name in ['1A2','2A2'] else 3 if trial_name in ['1A3','2A3'] else 4
            all_ds = MbdDataset_singlechannel(mode, all_x1, all_x2, all_x3, all_x4, all_y, transforms, all_params)

        if trial_name in ['1A5','2A5']:
            all_ds = MbdDataset_4channel(all_x1, all_x2, all_x3, all_x4, all_y, transforms, all_params)
        
        all_loader = DataLoader(all_ds, batch_size=bs, shuffle=False, num_workers=10) # here no shuffle

        for batch_data in all_loader:

            if trial_name in ['1A1','1A2','1A3','1A4', '2A1','2A2','2A3','2A4']:
                inputs, labels, _params = batch_data[0].to(device), batch_data[1].to(device), batch_data[2]#.to(device) # (5, 2, 768, 768) || nn.Conv2d expects an input of [batch_size, channels, height, width]
            
            if trial_name in ['1A5','2A5']:
                inputs, labels, _params = torch.cat((batch_data[0],batch_data[1],batch_data[2],batch_data[3]), dim=1).to(device), \
                    batch_data[4].to(device), batch_data[5]#.to(device) # (5, 4, 768, 768) || nn.Conv2d expects an input of [batch_size, channels, height, width]
            _params = _params.type(torch.LongTensor).to(device) # this format disrupt age, weight, height - making them integers

            outputs = model(inputs) # [bs, 256]
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, labels], dim=0)
            _params_list = torch.cat([_params_list, _params], dim=0)

    y = y.detach().cpu().numpy().reshape((len(y),1))
    y_pred = y_pred.detach().cpu().numpy()
    _params_list = _params_list.detach().cpu().numpy()

    results = pd.DataFrame(np.hstack((y_pred, _params_list, y)), 
                        columns = [f'v{i}' for i in range(int(dim))] + image_params.columns.tolist() + ['fragility_fracture'])

    results.to_csv(f'../data/embeds/{trial_name}_fold{fold}_dim{dim}.csv')
    print(f'Done with {trial_name}_fold{fold}_dim{dim}')

# python gen_embeds.py 1A1 
# python gen_embeds.py 1A2
# python gen_embeds.py 1A3 
# python gen_embeds.py 1A4 
# python gen_embeds.py 1A5

