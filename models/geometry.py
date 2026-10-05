import torch
from torch_scatter import scatter_add


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

def debug_user_eq_transform(score_d, pos, edge_index, edge_length):
    
    debug_info = {} 

    debug_info['input_score_d'] = score_d.clone().detach()
    debug_info['input_pos'] = pos.clone().detach()
    debug_info['input_edge_index'] = edge_index.clone().detach()
    debug_info['input_edge_length'] = edge_length.clone().detach()

    if (edge_length == 0).any():
        debug_info['log_message'] = "輸入的 edge_length 中包含零"
    elif (edge_length < 1e-9).any():
         debug_info['log_message'] = "輸入的 edge_length 中包含極小的數值 (< 1e-9)，可能導致數值不穩定。"
    else:
         debug_info['log_message'] = "輸入的 edge_length 檢查通過"

    N = pos.size(0)
    epsilon = 1e-6

    one_div_edge_length = 1. / edge_length
    debug_info['step1_one_div_edge_length'] = one_div_edge_length.clone().detach()

    term1 = one_div_edge_length + epsilon
    debug_info['step2_one_div_edge_length_plus_epsilon'] = term1.clone().detach()

    term2 = (pos[edge_index[0]] - pos[edge_index[1]])
    debug_info['step3_pos_difference_vector'] = term2.clone().detach()
    
    dd_dr = term1.view(-1, 1) * term2
    debug_info['step4_final_dd_dr'] = dd_dr.clone().detach()

    force_term = dd_dr * score_d.view(-1, 1)
    debug_info['step5_force_term (dd_dr * score_d)'] = force_term.clone().detach()

    scatter_term1 = scatter_add(force_term, edge_index[0], dim=0, dim_size=N)
    debug_info['step6_scatter_add_on_atom0'] = scatter_term1.clone().detach()

    scatter_term2 = scatter_add(-force_term, edge_index[1], dim=0, dim_size=N)
    debug_info['step7_scatter_add_on_atom1'] = scatter_term2.clone().detach()

    score_pos = scatter_term1 + scatter_term2
    debug_info['step8_final_score_pos'] = score_pos.clone().detach()
    
    if torch.isnan(score_pos).any():
        debug_info['final_log_message'] = "最終的 score_pos 包含 NaN"

    return score_pos, debug_info

# 將edge的訊息(貢獻度)整合到node上，會是(E, 3)
def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (E, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
        + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    return score_pos

# soft inverse version -> smooth
def eq_transform_smooth(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    eps = 1e-2
    delta = pos[edge_index[0]] - pos[edge_index[1]]
    safe_r = torch.sqrt(edge_length ** 2 + eps ** 2)
    dd_dr = delta / safe_r
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
        + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length)  # (C, 3)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos


def get_angle(pos, angle_index):
    """
    Args:
        pos:  (N, 3)
        angle_index:  (3, A), left-center-right.
    """
    n1, ctr, n2 = angle_index   # (A, )
    v1 = pos[n1] - pos[ctr] # (A, 3)
    v2 = pos[n2] - pos[ctr]
    inner_prod = torch.sum(v1 * v2, dim=-1, keepdim=True)   # (A, 1)
    length_prod = torch.norm(v1, dim=-1, keepdim=True) * torch.norm(v2, dim=-1, keepdim=True)   # (A, 1)
    angle = torch.acos(inner_prod / length_prod)    # (A, 1)
    return angle


def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    """
    n1, ctr1, ctr2, n2 = dihedral_index # (A, )
    v_ctr = pos[ctr2] - pos[ctr1]   # (A, 3)
    v1 = pos[n1] - pos[ctr1]
    v2 = pos[n2] - pos[ctr2]
    n1 = torch.cross(v_ctr, v1, dim=-1) # Normal vectors of the two planes
    n2 = torch.cross(v_ctr, v2, dim=-1)
    inner_prod = torch.sum(n1 * n2, dim=1, keepdim=True)    # (A, 1)
    length_prod = torch.norm(n1, dim=-1, keepdim=True) * torch.norm(n2, dim=-1, keepdim=True)
    dihedral = torch.acos(inner_prod / length_prod)
    return dihedral


