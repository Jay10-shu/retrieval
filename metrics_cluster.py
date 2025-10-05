from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F

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


# def clusterSim_jsd(v_alpha, t_alpha):
#     batch_size = v_alpha.shape[0]     # N (视频数)
#     batch_size_t = t_alpha.shape[0] # M (文本数)
#     # 使数据顺序变为 (t1,v1), (t1,v2)...(t1,vN), (t2,v1), (t2,v2)...
#     v_alpha = v_alpha.repeat(batch_size_t, 1)                  # 扩展为 [M*N, C]
#     t_alpha = t_alpha.repeat_interleave(batch_size, dim=0)     # 扩展为 [M*N, C]
#     # JSD 向量的内在顺序现在是“文本优先”，天然对应 [M, N] 矩阵
#     JSD = 0.5*compute_KL(v_alpha, 0.5*(v_alpha + t_alpha)) + 0.5*compute_KL(t_alpha, 0.5*(v_alpha + t_alpha))
#     # JSD_kl= 0.5*compute_KL(v_alpha,t_alpha) + 0.5*compute_KL(t_alpha, v_alpha )
#     # 将 [文本, 视频] 顺序的数据，填入 [文本, 视频] 形状的矩阵
#     JSD = JSD.view(batch_size_t, batch_size) # -> 维度为 [M, N]
#     Sim = -JSD
#     return Sim
# def compute_KL(v_alpha,t_alpha):
#     epsilon = 1e-8
#     v_alpha = v_alpha + epsilon
#     t_alpha = t_alpha + epsilon
#     # Compute components of KL divergence
#     sum_v_alpha = v_alpha.sum(dim=-1)
#     sum_t_alpha = t_alpha.sum(dim=-1)
#     # First term
#     first = ((torch.lgamma(t_alpha)).sum(dim=-1)) + (torch.lgamma(sum_v_alpha)) - (torch.lgamma(sum_t_alpha)) - ((torch.lgamma(v_alpha)).sum(dim=-1))
#     second = ((v_alpha - t_alpha) * (torch.digamma(v_alpha) - torch.digamma(sum_v_alpha)[:, None])).sum(dim=-1)
#     # Total KL divergence
#     loss = first + second
#     return loss
 


# def clusterSim_jsd(v_alpha, t_alpha):
#     batch_size = v_alpha.shape[0]     # N (视频数)
#     batch_size_t = t_alpha.shape[0] # M (文本数)
#     v_alpha_expanded = v_alpha.repeat(batch_size_t, 1)
#     t_alpha_expanded = t_alpha.repeat_interleave(batch_size, dim=0)
#     # --- 核心修改：从alpha参数计算出期望的概率分布 ---
#     # 1. 计算视频的期望概率分布 p_v
#     v_prob = v_alpha_expanded / torch.sum(v_alpha_expanded, dim=1, keepdim=True)
#     # 2. 计算文本的期望概率分布 p_t
#     t_prob = t_alpha_expanded / torch.sum(t_alpha_expanded, dim=1, keepdim=True)
#     # 3. 计算混合分布 M
#     m_prob = 0.5 * (v_prob + t_prob)
#     # --- 4. 使用标准的KL散度计算JSD ---
#     #    F.kl_div(target.log(), input, ...) 计算 KL(input || target)
#     #    我们需要计算 KL(v_prob || m_prob) 和 KL(t_prob || m_prob)
#     #    注意PyTorch的kl_div期望target是log-probabilities
#     jsd = 0.5 * F.kl_div(m_prob.log(), v_prob, reduction='none').sum(dim=1) + \
#             0.5 * F.kl_div(m_prob.log(), t_prob, reduction='none').sum(dim=1)
#     jsd = jsd.view(batch_size_t, batch_size) # -> 维度为 [M, N]
#     sim = -jsd # 使用我们之前讨论的、更稳定的负距离作为相似度
#     return sim



def clusterSim_jsd(v_alpha, t_alpha):
    """
    Calculates a similarity matrix based on the negative JS Divergence
    between video and text Dirichlet distributions.
    Args:
        v_alpha: Video alpha parameters, shape [N, K] (N videos, K clusters)
        t_alpha: Text alpha parameters, shape [M, K] (M texts, K clusters)
    Returns:
        Similarity matrix, shape [M, N]
    """
    num_videos = v_alpha.shape[0]
    num_texts = t_alpha.shape[0]
    # Create all-pairs for batch computation
    # v_alpha_expanded: [M*N, K]
    # t_alpha_expanded: [M*N, K]
    v_alpha_expanded = v_alpha.unsqueeze(0).repeat(num_texts, 1, 1).view(-1, v_alpha.shape[1])
    t_alpha_expanded = t_alpha.unsqueeze(1).repeat(1, num_videos, 1).view(-1, t_alpha.shape[1])
    # 1. Calculate the alpha parameters of the mixture distribution M
    m_alpha = 0.5 * (v_alpha_expanded + t_alpha_expanded)    
    # 2. Calculate the two KL divergence terms using the correct formula
    kl_v_m = kl_divergence_dirichlet(v_alpha_expanded, m_alpha)
    kl_t_m = kl_divergence_dirichlet(t_alpha_expanded, m_alpha)    
    # 3. Combine into JS Divergence
    jsd = 0.5 * (kl_v_m + kl_t_m)    
    # Reshape into a matrix
    jsd_matrix = jsd.view(num_texts, num_videos) # -> shape [M, N]    
    # Use negative distance as similarity
    sim_matrix = -jsd_matrix     
    return sim_matrix

def kl_divergence_dirichlet(alpha, beta):
    """
    Calculates the KL divergence between two Dirichlet distributions.
    KL(Dir(alpha) || Dir(beta))
    
    Args:
        alpha: Parameters of the first distribution, shape [N, K]
        beta: Parameters of the second distribution, shape [N, K]
    
    Returns:
        KL divergence, shape [N]
    """
    # Ensure parameters are on the same device
    beta = beta.to(alpha.device)
    sum_alpha = torch.sum(alpha, dim=1)
    sum_beta = torch.sum(beta, dim=1)
    # Log of Gamma functions
    term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
    # Sum of log of Gamma functions for individual parameters
    term2 = torch.sum(torch.lgamma(beta) - torch.lgamma(alpha), dim=1)     
    # Digamma functions part
    alpha_minus_beta = alpha - beta
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha).unsqueeze(1).expand_as(alpha)     
    term3 = torch.sum(alpha_minus_beta * (digamma_alpha - digamma_sum_alpha), dim=1)    
    return term1 + term2 + term3
