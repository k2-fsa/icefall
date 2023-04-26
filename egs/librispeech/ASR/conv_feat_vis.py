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

feats_all = feats_all.transpose(0,1)
print(feats_all.shape)
exit()

for i in range(512):
    plt.hist(feats_all[i])
    plt.savefig(f'./conv_feat/dim_{i}.pdf')
    plt.close()
