import torch
import numpy as np

checkpoint = torch.load("./trained_models/GPT2_M_slimming/e2e/model.52575.pt", map_location="cpu")['model_state_dict']
records = []
count = 0
print(checkpoint.keys())
for k in checkpoint:
    if 'slimming_coef' in k:
        records.append(checkpoint[k].detach().view(-1).numpy())
        count += 1

print(count)
np.save('self_slimming_coef_records.npy', records)