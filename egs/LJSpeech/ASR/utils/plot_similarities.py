import os
import torch
import matplotlib.pyplot as plt

out_dir='outputs/'

org = torch.load('pruned_transducer_stateless_d2v_v2/d2v-T/epoch-27.pt')
ft = torch.load('pruned_transducer_stateless_d2v_v2/d2v-T-LJft-oracle-epoch100/best-valid-loss.pt')
pft = torch.load('pruned_transducer_stateless_d2v_v2/d2v-T-LJft-pseudo-trained/best-valid-loss.pt')

org_state_dict, ft_state_dict, pft_state_dict = org['model'], ft['model'], pft['model']

# # Define three model parameters
# for (n1, p1), (n2, p2), (n3, p3) in zip(org.named_parameters(), ft.named_parameters(), pft.named_parameters()):
#     assert n1 == n2 == n3
#     org_state_dict[n1] = p1
#     ft_state_dict[n1] = p2
#     pft_state_dict[n1] = p3

for name in org_state_dict.keys():
    # Define the weight names
    weights = [org_state_dict[name], ft_state_dict[name], pft_state_dict[name]]

    # Define the x-axis labels
    x = ['org vs ft', 'org vs pft', 'ft vs pft']
    abs_diff, rel_diff, cos_sim = [], [], []
    
    i = 0
    k = 0
    for j in range(i+1,3):
        if weights[i].dim() > 1:
            wi = weights[i].view(-1)
            wj = weights[j].view(-1)
        else:
            wi = weights[i]
            wj = weights[j]
        
        # Compute absolute difference
        abs_diff.append(torch.abs(wi - wj).sum(-1).cpu().numpy())

        # Compute relative difference
        rel_diff.append(abs_diff[k] / torch.max(torch.abs(wi).sum(-1), torch.abs(wj).sum(-1)).cpu().numpy())

        # Compute cosine similarity
        cos_sim.append(torch.nn.functional.cosine_similarity(wi, wj, dim=0).cpu().numpy())

        k += 1
    
    i, j = 1, 2
    if weights[i].dim() > 1:
        wi = weights[i].view(-1)
        wj = weights[j].view(-1)
    else:
        wi = weights[i]
        wj = weights[j]

    # Compute absolute difference
    abs_diff.append(torch.abs(wi - wj).sum(-1).cpu().numpy())

    # Compute relative difference
    rel_diff.append(abs_diff[2] / torch.max(torch.abs(wi).sum(-1), torch.abs(wj).sum(-1)).cpu().numpy())

    # Compute cosine similarity
    cos_sim.append(torch.nn.functional.cosine_similarity(wi, wj, dim=0).cpu().numpy())

    if abs_diff[1] > 0.001:
        print(name)

        for k, metric in enumerate([abs_diff, rel_diff, cos_sim]):
            y = metric
            # Plot results as a bar graph
            plt.bar(x, y)

            if not os.path.exists(out_dir + name):
                os.makedirs(out_dir + name, exist_ok=True)
            plt.savefig(out_dir + name + "/" + ['abs_diff', 'rel_diff', 'cos_sim'][k] + ".png")
            plt.close()