import torch
import copy
from torch.utils.data import Dataset


class PPODataset(Dataset):

    def __init__(self, batch_size, minibatch_size, is_discrete, device):

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict
        if self.values_dict is not None:
            self.length = (self.values_dict['outs'] == 0).sum() // self.minibatch_size

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]	           
        end = self.last_range[1]	
        self.values_dict['mu'][start:end] = mu	
        self.values_dict['sigma'][start:end] = sigma 

    def __len__(self):
        return self.length
    
    def clear_out_infos(self):
        outs = self.values_dict.get('outs')
        if outs is None:
            return

        zero_mask = (outs.view(-1) == 0)

        for k, v in self.values_dict.items():
            if v is None:
                continue
            if isinstance(v, dict):
                self.values_dict[k] = {
                    sub_k: sub_v[zero_mask] for sub_k, sub_v in v.items()
                }
            else:
                self.values_dict[k] = v[zero_mask]
        

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k,v in self.values_dict.items():
            if v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[start:end] for kd, vd in v.items() }
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[start:end]
                
        return input_dict

    def __getitem__(self, idx):
        sample = self._get_item(idx)
        return sample



class DatasetList(Dataset):
    def __init__(self):
        self.dataset_list = []

    def __len__(self):
        return self.dataset_list[0].length * len(self.dataset_list)

    def add_dataset(self, dataset):
        self.dataset_list.append(copy.deepcopy(dataset))

    def clear(self):
        self.dataset_list = []

    def __getitem__(self, idx):
        ds_len = len(self.dataset_list)
        ds_idx = idx % ds_len
        in_idx = idx // ds_len
        return self.dataset_list[ds_idx].__getitem__(in_idx)