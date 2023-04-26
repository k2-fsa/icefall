from glob import glob
import numpy as np
import matplotlib.pyplot as plt

feats = []
feat_list = glob('./conv_feat/*')
for feat in feat_list:
    feat = np.load(feat)
    feats.append(feat)

feats_all = feats[0]
for feat in feats:
    feats_all = np.concatenate([feats_all, feat])

feats_all = feats_all.transpose(0,1)

for i in range(512):
    plt.hist(feats_all[i])
    plt.savefig(f'dim_{i}')
    plt.close()
