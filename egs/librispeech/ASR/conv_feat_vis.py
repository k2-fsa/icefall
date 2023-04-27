from glob import glob
import numpy as np
import matplotlib.pyplot as plt

feats = []
feat_list = glob('./conv_feat/*.npy')
for feat in feat_list:
    feat = np.load(feat)
    feats.append(feat)

feats_all = feats[0]
for feat in feats:
    feats_all = np.concatenate([feats_all, feat])

feats_all = feats_all.transpose(1,0)
print(feats_all.shape)

for i in range(512):
    mean = feats_all[i].mean()
    std = feats_all[i].std()
    print(mean, std)
'''
for i in range(512):
    plt.hist(feats_all[i], bins=500)
    plt.savefig(f'./conv_feat/dim_{i}.pdf')
    plt.close()
'''