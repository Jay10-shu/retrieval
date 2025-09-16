from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import torch


def compute_metrics(x):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.argmax(ind == 0, axis=1)
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics,ind


def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10]):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = (valid_ranks + 1).float().mean().item()
    results["Std_Rank"] = (valid_ranks + 1).float().std().item()
    results['MR'] = results["MedianR"]
    return results

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T



def get_max_indices(a,b,top_k=10):
    if torch.is_tensor(a):
        a = a.cpu().numpy()
    if torch.is_tensor(b):
        a = b.cpu().numpy()
    diff = a - b
    zero_indices_in_b = np.where(b == 0)[0]
    filtered_diffs = diff[zero_indices_in_b]
    num_top_values = min(top_k, len(filtered_diffs))
    top_diffs_indices_in_filtered = np.argsort(filtered_diffs)[-num_top_values:][::-1]
    top_diffs_indices_in_a = zero_indices_in_b[top_diffs_indices_in_filtered]
    top_diffs_values = filtered_diffs[top_diffs_indices_in_filtered]
    return top_diffs_values, top_diffs_indices_in_a


def clusterSim_jsd(v_alpha, t_alpha):
    batch_size = v_alpha.shape[0]
    batch_size_t = t_alpha.shape[0]
    v_alpha = v_alpha.repeat_interleave(batch_size_t, dim=0)
    t_alpha = t_alpha.repeat(batch_size,1)
    JSD = 0.5*compute_KL(v_alpha, 0.5*(v_alpha + t_alpha)) + 0.5*compute_KL(t_alpha, 0.5*(v_alpha + t_alpha))
    JSD = JSD.view(batch_size, batch_size_t)
    Sim = 1-JSD
    return Sim
   
def compute_KL(v_alpha,t_alpha):
    epsilon = 1e-8
    v_alpha = v_alpha + epsilon
    t_alpha = t_alpha + epsilon

    # Compute components of KL divergence
    sum_v_alpha = v_alpha.sum(dim=-1)
    sum_t_alpha = t_alpha.sum(dim=-1)
    
    # First term
    first = ((torch.lgamma(t_alpha)).sum(dim=-1)) + (torch.lgamma(sum_v_alpha)) - (torch.lgamma(sum_t_alpha)) - ((torch.lgamma(v_alpha)).sum(dim=-1))
    second = ((v_alpha - t_alpha) * (torch.digamma(v_alpha) - torch.digamma(sum_v_alpha)[:, None])).sum(dim=-1)
    
    # Total KL divergence
    loss = first + second
    return loss




