import torch

base_model = torch.load('./d2v-base-T.pt')
bias_model = torch.load('./bitfit_2414_v2/checkpoint-100.pt')

base_model, bias_model = base_model['model'], bias_model['model']

for key in base_model.keys():
    if 'bias' in key:
        l1_diff = torch.abs(base_model[key]-bias_model[key]).sum()
        print(key, l1_diff)
