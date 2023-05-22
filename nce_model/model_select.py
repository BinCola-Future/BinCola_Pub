# coding:utf-8
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE
import numpy as np
from nce_model.SelfAttention import ScaledDotProductAttention



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, att_type, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.att_type = att_type
        self.d_model = d_model
        self.slf_attn = ScaledDotProductAttention(d_model=self.d_model, d_k=d_k, d_v=d_v, h=n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        if self.att_type == 'SelfAttention':
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attention_mask=slf_attn_mask)
        elif self.att_type == 'NoAttention':
            enc_output = enc_input
            enc_slf_attn = None
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class SiameseAttentionNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_layers, n_head, d_k, d_v, att_type, dropout, out_type):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_type = out_type
        self.layer_stack = nn.ModuleList([
            EncoderLayer(feature_dim, hidden_dim, n_head, d_k, d_v, att_type, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src, pos, neg):
        attn_list = []
        pos_num = pos.shape[1]
        neg_num = neg.shape[1]
        src = src.unsqueeze(-1)
        pos = pos.unsqueeze(-1)
        neg = neg.unsqueeze(-1)
        pad = (self.feature_dim-1,0)
        output_src = F.pad(src,pad,'constant',0)
        output_pos = F.pad(pos,pad,'constant',0)
        output_neg = F.pad(neg,pad,'constant',0)

        for enc_layer in self.layer_stack:
            output_src, slf_attn = enc_layer(output_src, slf_attn_mask=None)
            if slf_attn != None:
                attn_list += [slf_attn]

            for idx in range(pos_num):
                output_pos[:,idx,:,:], _ = enc_layer(output_pos[:,idx,:,:], slf_attn_mask=None)

            for idx in range(neg_num):
                output_neg[:,idx,:,:], _ = enc_layer(output_neg[:,idx,:,:], slf_attn_mask=None)
            
        if self.out_type == 'mean':
            output_src = output_src.mean(dim=-2)
            output_pos = output_pos.mean(dim=-2)
            output_neg = output_neg.mean(dim=-2)
        elif self.out_type == 'sum':
            output_src = output_src.sum(dim=-2)
            output_pos = output_pos.sum(dim=-2)
            output_neg = output_neg.sum(dim=-2)
        elif self.out_type == 'last':
            output_src = output_src[:,-1,:]
            output_pos = output_pos[:,:,-1,:]
            output_neg = output_neg[:,:,-1,:]
        similarity = F.cosine_similarity(torch.cat((output_src,output_src),0), torch.cat((output_pos[:,0,:],output_neg[:,0,:]),0), dim=-1, eps=1e-8)
        return output_src, output_pos, output_neg, similarity, attn_list

    def data_normal(self, origin_data):
        d_min = origin_data.min()
        if d_min < 0:
            origin_data += torch.abs(d_min)
            d_min = origin_data.min()
        d_max = origin_data.max()
        dst = d_max - d_min
        norm_data = (origin_data - d_min).true_divide(dst)
        return norm_data

class UseAttentionNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_layers, n_head, d_k, d_v, att_type, dropout, out_type):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_type = out_type
        self.layer_stack = nn.ModuleList([
            EncoderLayer(feature_dim, hidden_dim, n_head, d_k, d_v, att_type, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src):
        attn_list = []
        src = src.unsqueeze(-1)
        pad = (self.feature_dim-1,0)
        output_src = F.pad(src,pad,'constant',0)

        for enc_layer in self.layer_stack:
            output_src, slf_attn = enc_layer(output_src, slf_attn_mask=None)
            if slf_attn != None:
                attn_list += [slf_attn]

        if self.out_type == 'mean':
            output_src = output_src.mean(dim=-2)
        elif self.out_type == 'sum':
            output_src = output_src.sum(dim=-2)
        elif self.out_type == 'last':
            output_src = output_src[:,-1,:]
        return output_src, attn_list

    def data_normal(self, origin_data):
        d_min = origin_data.min()
        if d_min < 0:
            origin_data += torch.abs(d_min)
            d_min = origin_data.min()
        d_max = origin_data.max()
        dst = d_max - d_min
        norm_data = (origin_data - d_min).true_divide(dst)
        return norm_data


# InfoNCE
def InfoNCELoss(query, positive_key, negative_keys, temper):
    criterion = InfoNCE(temperature=temper,negative_mode='paired')
    loss = criterion(query, positive_key, negative_keys)
    return loss

