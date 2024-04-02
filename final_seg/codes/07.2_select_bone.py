# crop large bone, then pad to (1320,1320) \\ cuz largest w | h is 1318
from utils import *
import sys
# import gc

os.chdir('/home/chenfy/projects/seg/codes')

def project_back(x):
    x = int(x * 1844/512 + 230)
    return x

sel = pd.read_csv('../data/meta/selected_seg_meta.csv', index_col=0)
name = sys.argv[1]

pt, scan = name.split('_')

if not os.path.isfile(f'/nfs/public/fuwai_PQCT/largebone_npy/{pt}/{scan}.npy'):

    fpath = f'/nfs/public/fuwai_PQCT/img/{pt}/{scan}.pkl'
    img = pkl.load(open(f'{fpath}', 'rb'))[80,:,:].numpy()

    lower_1, upper_1, left_1, right_1,  lower_2, upper_2, left_2, right_2 = sel.loc[name,['0','1','2','3','4','5','6','7']]
    
    crop_largebone = img[project_back(lower_1)-10:project_back(upper_1)+10,project_back(left_1)-10:project_back(right_1)+10] # pad in transformation in later training
    # extra_w_oneside = (1320 - (project_back(left_1)-project_back(right_1)))/2
    # extra_h_oneside = (1320 - (project_back(upper_1)-project_back(lower_1)))/2
    # pad_largebone = np.pad(crop_largebone, ((extra_h_oneside, extra_h_oneside), (extra_w_oneside, extra_w_oneside)),
    #     mode='constant', constant_values=0) # ((extra_top, extra_bottom), (extra_left, extra_right))
    
    crop_smallbone = img[project_back(lower_2)-10:project_back(upper_2)+10, project_back(left_2)-10:project_back(right_2)+10]

    if not os.path.isdir(f'/nfs/public/fuwai_PQCT/largebone_npy/{pt}'):
        os.system(f'mkdir /nfs/public/fuwai_PQCT/largebone_npy/{pt}')
    if not os.path.isdir(f'/nfs/public/fuwai_PQCT/smallbone_npy/{pt}'):
        os.system(f'mkdir /nfs/public/fuwai_PQCT/smallbone_npy/{pt}')
    np.save(f'/nfs/public/fuwai_PQCT/largebone_npy/{pt}/{scan}.npy', crop_largebone)
    np.save(f'/nfs/public/fuwai_PQCT/smallbone_npy/{pt}/{scan}.npy', crop_smallbone)


    plt.imshow(crop_largebone, cmap='Greys')
    plt.savefig(f'../images/largebone2d/{pt}_{scan}.jpg', dpi=300)

    plt.imshow(crop_smallbone, cmap='Greys')
    plt.savefig(f'../images/smallbone2d/{pt}_{scan}.jpg', dpi=300)

    # gc.collect() # to empty cache

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print(f'Done converting {pt}_{scan}  {time}')



# tmux a -t 0
# python select_bone.py > ../logs/020223_selectbone.log # program slowed down with time ?? so I added gc.collect()
# python select_bone.py > ../logs/020323_selectbone.log # still very slow
# 
    