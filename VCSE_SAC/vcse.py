import torch
import pandas as pd
import torch.nn as nn
import utils
import torch.nn.functional as F
import bisect
import math

class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward

    
    
class VCSE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, knn_k,device):
        self.knn_k = knn_k
        self.device = device

    def __call__(self, state,value):
        #value => [b1 , 1]
        #state => [b1 , c]
        #z => [b1, c+1]
        # [b1] => [b1,b1]
        ds = state.size(1)
        source = target = state
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c+1) - (1, b2, c+1) -> (b1, 1, c+1) - (1, b2, c+1) -> (b1, b2, c+1) -> (b1, b2)
        sim_matrix_s = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)

        source = target = value
        # (b1, 1, 1) - (1, b2, 1) -> (b1, 1, 1) - (1, b2, 1) -> (b1, b2, 1) -> (b1, b2)
        sim_matrix_v = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        
        sim_matrix = torch.max(torch.cat((sim_matrix_s.unsqueeze(-1),sim_matrix_v.unsqueeze(-1)),dim=-1),dim=-1)[0]
        eps, index = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        state_norm, index = sim_matrix_s.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        value_norm, index = sim_matrix_v.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        eps = eps[:, -1] #k-th nearest distance
        eps = eps.reshape(-1, 1) # (b1, 1)
        
        state_norm = state_norm[:, -1] #k-th nearest distance
        state_norm = state_norm.reshape(-1, 1) # (b1, 1)

        value_norm = value_norm[:, -1] #k-th nearest distance
        value_norm = value_norm.reshape(-1, 1) # (b1, 1)
        
        sim_matrix_v = sim_matrix_v < eps
        n_v = torch.sum(sim_matrix_v,dim=1,keepdim = True) # (b1,1)
        
        sim_matrix_s = sim_matrix_s < eps
        n_s = torch.sum(sim_matrix_s,dim=1,keepdim = True) # (b1,1)        
        reward = torch.digamma((n_v+1).to(torch.float)) / ds + torch.log(eps * 2 + 0.00001)
        return reward, n_v,n_s, eps, state_norm, value_norm