# coding:utf-8
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset


"""
tp_pairs:[[{func_src:feature},[{tp_func_dst:feature}]]]
tn_pairs:[[{func_src:feature},[{tn_func_dst:feature}]]]
"""
def tp_pairs_parse(pairs):
    src_funcs_option = []
    src_funcs_feature = []
    pos_funcs_options = []
    pos_funcs_features = []
    for pair in pairs:
        pair_src = pair[0]
        src_funcs_option.append(list(pair_src.keys())[0])
        src_funcs_feature.append(list(pair_src.values())[0])
        pair_dsts = pair[1]
        pos_funcs_option = []
        pos_funcs_feature = []
        for pair_dst in pair_dsts:
            pos_funcs_option.append(list(pair_dst.keys())[0])
            pos_funcs_feature.append(list(pair_dst.values())[0])
        pos_funcs_options.append(pos_funcs_option)
        pos_funcs_features.append(pos_funcs_feature)
    return src_funcs_option,src_funcs_feature,pos_funcs_options,pos_funcs_features


def tn_pairs_parse(pairs):
    neg_funcs_options = []
    neg_funcs_features = []
    for pair in pairs:
        pair_dsts = pair[1]
        neg_funcs_option = []
        neg_funcs_feature = []
        for pair_dst in pair_dsts:
            neg_funcs_option.append(list(pair_dst.keys())[0])
            neg_funcs_feature.append(list(pair_dst.values())[0])
        neg_funcs_options.append(neg_funcs_option)
        neg_funcs_features.append(neg_funcs_feature)
    return neg_funcs_options,neg_funcs_features
            


class MyDataset(Dataset):
    def __init__(self, tp_pairs, tn_pairs, device):

        src_funcs_option,src_funcs_feature,pos_funcs_options,pos_funcs_features = tp_pairs_parse(tp_pairs)
        neg_funcs_options,neg_funcs_features = tn_pairs_parse(tn_pairs)


        self.src_option = src_funcs_option
        self.src = torch.from_numpy(np.array(src_funcs_feature)).float().to(device)
        self.pos = torch.from_numpy(np.array(pos_funcs_features)).float().to(device)
        self.neg = torch.from_numpy(np.array(neg_funcs_features)).float().to(device)


    def __getitem__(self, index):
        return self.src_option[index],self.src[index],self.pos[index],self.neg[index]

    def __len__(self):
        return len(self.src_option)


def CreateDataLoader(tp_pairs, tn_pairs, batch_size, device):
    dataset = MyDataset(tp_pairs, tn_pairs, device)
    data_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return data_loader