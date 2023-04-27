from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

spk_list = sorted(glob('./conv_feat/*'))
for spk in tqdm(spk_list):
    spk_id = spk.split('/')[-1]
    feats = []
    feat_list = glob(f'{spk}/*.npy')
    static_file = open(f'{spk}/{spk_id}_statistic.txt', 'w')

    for feat in tqdm(feat_list, leave=False):
        feat = np.load(feat)
        feats.append(feat)

    feats_all = feats[0]
    for feat in feats:
        feats_all = np.concatenate([feats_all, feat])

    feats_all = feats_all.transpose(1,0)

    for i in tqdm(range(512), leave=False):
        mean = feats_all[i].mean()
        std = feats_all[i].std()
        static_file.write(f'{mean} {std}')
    '''
    for i in range(512):
        plt.hist(feats_all[i], bins=500)
        plt.savefig(f'./conv_feat/dim_{i}.pdf')
        plt.close()
    '''
