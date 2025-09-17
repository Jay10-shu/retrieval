from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.special as sp  # For digamma and gammaln functions
from timm.models.layers import DropPath

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


 

class JSDivergence(nn.Module):

    def __init__(self):
        super(JSDivergence, self).__init__()    
    def forward(self, v_alpha, t_alpha):
        batch_size = v_alpha.shape[0]
        batch_size_t = t_alpha.shape[0]
        v_alpha = v_alpha.repeat(batch_size_t, 1)                  # 扩展为 [M*N, C]
        t_alpha = t_alpha.repeat_interleave(batch_size, dim=0)     # 扩展为 [M*N, C]
        jsd = 0.5*self.compute_KL(v_alpha, 0.5*(v_alpha+t_alpha)) + 0.5*self.compute_KL(t_alpha, 0.5*(v_alpha+t_alpha))
        jsd = jsd.view(batch_size_t, batch_size)
        sim = 1-jsd
        return sim

    def compute_KL(self, v_alpha, t_alpha):
        # Add small constant for numerical stability
        epsilon = 1e-8
        v_alpha = v_alpha + epsilon
        t_alpha = t_alpha + epsilon
        sum_v_alpha = v_alpha.sum(dim=-1)
        sum_t_alpha = t_alpha.sum(dim=-1)
        first = ((torch.lgamma(t_alpha)).sum(dim=-1)) + (torch.lgamma(sum_v_alpha)) - (torch.lgamma(sum_t_alpha)) - ((torch.lgamma(v_alpha)).sum(dim=-1))
        second = ((v_alpha - t_alpha) * (torch.digamma(v_alpha) - torch.digamma(sum_v_alpha)[:, None])).sum(dim=-1)
        # Total KL divergence
        loss = first + second
        return loss
    
    

