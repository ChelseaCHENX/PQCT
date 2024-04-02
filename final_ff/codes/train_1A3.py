trial_name = '1A3'
class_out = 2
dim_hidden = 128 # size of embeds
# n_input_channels=1
# self.pqcts = torch.nn.Linear(num_classes,13)

import monai
import os
root_dir = '/home/chenfy/projects/classifier_fragilefracture'
os.chdir(f'{root_dir}/codes')

if not os.path.isdir(f'{root_dir}/data/model/{trial_name}'):
    os.mkdir(f'{root_dir}/data/model/{trial_name}')
    
from utils import *

from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import f1_score

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

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=class_out)]) # return (num_class, num_samples), as binary results
auc_metric = ROCAUCMetric()

# filelist
meta = pd.read_csv('../data/meta/chivos_fragfrax_final.csv')#.iloc[:100,:] # pt_scan, 0-5

## scale data to adjust losses
meta['age'] = meta['age'] / 100
meta['height'] = meta['height'] / 100 # original in cm
meta['weight'] = meta['weight'] / 10 # original in kg

image_files_list1, image_files_list2, image_files_list3, image_files_list4 = meta['tibia_path'], meta['fibula_path'], meta['radius_path'], meta['ulna_path']
for name in ['RADIUS_Tt_Ar', 'RADIUS_Tt_vBMD', 'RADIUS_Tb_Ar', 'RADIUS_Tb_vBMD', 'RADIUS_Ct_Ar','RADIUS_Ct_vBMD','RADIUS_Ct_Pm',  'TIBIA_Tt_Ar', 'TIBIA_Tt_vBMD', 'TIBIA_Tb_Ar', 'TIBIA_Tb_vBMD', 'TIBIA_Ct_Ar','TIBIA_Ct_vBMD','TIBIA_Ct_Pm']:
    meta[name] /= 100
for name in ['RADIUS_Ct_Po', 'TIBIA_Ct_Po']:
    meta[name] *= 100
image_label = meta['fragility_fracture']

image_params = meta[['sex','age','height','weight', 
                     'RADIUS_Tt_Ar','RADIUS_Tt_vBMD','RADIUS_Tb_Ar','RADIUS_Tb_vBMD','RADIUS_BV/TV','RADIUS_Tb_N','RADIUS_Tb_Th','RADIUS_Tb_Sp','RADIUS_Ct_Ar','RADIUS_Ct_vBMD','RADIUS_Ct_Pm','RADIUS_Ct_Po','RADIUS_Ct_Th']]

# define model
## inputs, labels, _params = batch_data[0].to(device), batch_data[1].to(device), batch_data[2]#.to(device) # (5, 4, 768, 768) || nn.Conv2d expects an input of [batch_size, channels, height, width]
## outputs, losses = model(inputs, labels, _params)
class CustomResNet(torch.nn.Module): 
    def __init__(self, num_classes, class_out): 
        super().__init__()
        net = resnet34(spatial_dims=2, n_input_channels=1, num_classes=num_classes) #----------- changed n_input_channels=1
        # self.dropout = 
        self.conv = torch.nn.Sequential(*list(net.children())[:-1])
        self.fc = net.fc

        self.classifier = torch.nn.Linear(num_classes,class_out) # MBD types, here is 2 || CrossEntropyLoss in PyTorch is already implemented with Softmax
        self.entropy = torch.nn.CrossEntropyLoss()

        self.sex = torch.nn.Linear(num_classes,2)
        self.entropy_bin = torch.nn.CrossEntropyLoss()

        self.agewtht = torch.nn.Linear(num_classes,3) # age, wt, ht
        self.mse_agewtht = torch.nn.MSELoss()

        self.pqcts = torch.nn.Linear(num_classes,13) # 13 pqct params for single image
        self.mse_pqct = torch.nn.MSELoss()

        self.activation = torch.nn.ReLU()
        
    def forward(self, x, labels, add_vars):
        
        # x = self.dropout(x) 
        x = self.conv(x) 
        x = x.view(-1, 512) # default resnet34 is 512 at this layer
        x = self.activation(self.fc(x)) # num_classes, eg, 256

        x,x1,x2,x3 = self.classifier(x), self.sex(x), self.agewtht(x), self.pqcts(x)

        loss0 = self.entropy(x, labels) # 4
        loss1 = self.entropy_bin(x1, add_vars[:,0]) # sex
        loss2 = self.mse_agewtht(x2, add_vars[:,[1,2,3]].reshape(x2.shape).to(torch.float32)) 
        loss3 = self.mse_pqct(x3, add_vars[:,4:].reshape(x3.shape).to(torch.float32)) 

        loss = loss0 + 0.01*loss1 + 0.01*loss2 + 0.1*loss3 
        return x,loss


# define transforms
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        SpatialPad(spatial_size=(768,768)),
        CenterSpatialCrop(roi_size=(768,768)),

        RandRotate(range_x=10, range_y=0, prob=.8), # for train_v2.py, change prob to 1 after augmentation
        RandFlip(prob=.5, spatial_axis=0),
        RandFlip(prob=.5, spatial_axis=1)
    ]
)

val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity(), SpatialPad(spatial_size=(768,768)), CenterSpatialCrop(roi_size=(768,768))])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=class_out)]) # return (num_class, num_samples), as binary results

# define dataset
class MbdDataset(torch.utils.data.Dataset):
    def __init__(self, image_files1, image_files2, image_files3, image_files4,   labels, transforms, params):
        self.image_files1 = image_files1 # tibia_path
        self.image_files2 = image_files2 # fibula_path
        self.image_files3 = image_files3 # radius_path
        self.image_files4 = image_files4 # ulna_path        
        self.labels = labels # n
        self.params = params # (n,17) 4+13=17: sex, age, ht, wt, pqcts
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transforms(self.image_files3[index]), self.labels[index], self.params[index,:] # self.transforms(self.image_files2[index]), self.transforms(self.image_files3[index]), self.transforms(self.image_files4[index]), self.labels[index], self.params[index,:]

# test set
test_frac = 0.1
lr = 1e-3
bs = 10
gradient_accumulation_steps = 3 # thus eq bs = 30
aug_ = 4 # 4

seed = 2023
np.random.seed(seed)
length = meta.shape[0]
indices = np.arange(length)
np.random.shuffle(indices)

indice_dict = {}
indice_dict['recoded_meta'] = meta
indice_dict['shuffled_indices'] = indices

test_split = int(test_frac * length) 
test_indices = indices[:test_split]
indice_dict['test_indices'] = test_indices

test_x1, test_x2, test_x3, test_x4 = [image_files_list1[i] for i in test_indices], [image_files_list2[i] for i in test_indices], [image_files_list3[i] for i in test_indices], [image_files_list4[i] for i in test_indices]
test_y = [image_label[i] for i in test_indices]
test_params = image_params.iloc[test_indices,:].values 

test_ds = MbdDataset(test_x1, test_x2, test_x3, test_x4, test_y, val_transforms, test_params)
test_loader = DataLoader(test_ds, batch_size=30, num_workers=10)

# model
max_epochs = 200 ##
val_interval = 1

# dev split => train + val (cross-validation)
k = 6
splits=KFold(n_splits=k,shuffle=True,random_state=seed)

for fold, (train_indices_tmp,val_indices_tmp) in enumerate(splits.split(indices[test_split:])):
    
    fold += 1
    # if fold > 1:
    #     break
    # start running from fold2
    # if fold == 1:
    #     continue

    train_indices, val_indices = indices[test_split:][train_indices_tmp], indices[test_split:][val_indices_tmp]
    train_seq = resample(train_indices, n_samples=len(train_indices)*aug_, random_state=2023) # augmentation

    indice_dict[f'train_indices_noaug_fold{fold}'] = train_indices
    indice_dict[f'val_indices_noaug_fold{fold}'] = val_indices
    indice_dict[f'train_indices_aug_fold{fold}'] = train_seq
    
    train_x1, train_x2, train_x3, train_x4 = [image_files_list1[i] for i in train_seq], [image_files_list2[i] for i in train_seq], [image_files_list3[i] for i in train_seq], [image_files_list4[i] for i in train_seq]
    train_y = [image_label[i] for i in train_seq]
    train_params = image_params.iloc[train_seq,:].values

    val_x1, val_x2, val_x3, val_x4 = [image_files_list1[i] for i in val_indices], [image_files_list2[i] for i in val_indices], [image_files_list3[i] for i in val_indices], [image_files_list4[i] for i in val_indices]
    val_y = [image_label[i] for i in val_indices]
    val_params = image_params.iloc[val_indices,:].values
    
    train_ds = MbdDataset(train_x1, train_x2, train_x3, train_x4, train_y, train_transforms, train_params)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=10)

    val_ds = MbdDataset(val_x1, val_x2,val_x3, val_x4, val_y, val_transforms, val_params)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=10)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # has to be the first device on list
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    model = CustomResNet(dim_hidden, class_out) 
    model = torch.nn.DataParallel(model, device_ids=[1,3]).to(device) # model.to(device) # model = 

    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)        
        for batch_data in train_loader:
                
            step += 1
            
            inputs, labels, _params = batch_data[0].to(device), batch_data[1].to(device), batch_data[2]#.to(device) # (5, 4, 768, 768) || nn.Conv2d expects an input of [batch_size, channels, height, width]
            _params = _params.type(torch.LongTensor).to(device)

            
            outputs, losses = model(inputs, labels, _params)
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, labels], dim=0) 

            loss = losses.sum() / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0: # accum 3 times before optimization, eq bs = bs * 3
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)

        y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc = auc_metric.aggregate()
        auc_metric.reset()
        del y_pred_act, y_onehot
        metric_values.append(auc)

        y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
        y = y.cpu().detach().numpy()
        f1 = f1_score(y, y_pred, labels=[0,1,2], average='macro')

        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print(
            f"{time}"
            f" fold{fold} - epoch{epoch + 1} average train loss: {epoch_loss:.4f}"
            f" train ROCAUC: {auc:.4f}"
            f" train accuracy: {acc_metric:.4f}"
            f" train f1: {f1:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            val_step = 0
            val_loss = 0

            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for batch_data in val_loader:

                    val_step += 1

                    inputs, labels, _params = batch_data[0].to(device), batch_data[1].to(device), batch_data[2]
                    _params = _params.type(torch.LongTensor).to(device)       

                    outputs,losses = model(inputs, labels, _params)
                    loss = losses.sum()
                    val_loss += loss.item()

                    y_pred = torch.cat([y_pred, outputs], dim=0)
                    y = torch.cat([y, labels], dim=0)

                val_loss /= val_step

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot

                y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
                y = y.cpu().detach().numpy()
                f1 = f1_score(y, y_pred, labels=[0,1,2], average='macro')

                result = auc

                print(
                f"{time}"
                f" fold{fold} - epoch{epoch + 1} average validation loss: {val_loss:.4f}"
                f" validation auc: {auc:.4f}"
                f" validation accuracy: {acc_metric:.4f}"
                f" validation f1: {f1:.4f}")
                
                if result > best_metric: # select best per val ROCAUC
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, f"data/model/{trial_name}/fold{fold}.pth")) # 2 class
                    print("saved new best metric model")
                    print(
                        f" current fold{fold}-epoch: {epoch + 1} current validation AUC: {acc_metric:.4f}"
                        f" best validation AUC: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )

    print(f"fold{fold} - train completed, best AUC: {best_metric:.4f} " f" at epoch: {best_metric_epoch}")

pkl.dump(indice_dict, open(f'../data/logs/indices_{trial_name}.pkl','wb'))
# python /home/chenfy/projects/classifier2/codes/train_1A1.py > /home/chenfy/projects/classifier2/logs/072223_1A1.log