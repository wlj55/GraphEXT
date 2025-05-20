import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
import numpy as np

def convert_edge_mask(masks, num_nodes, edge_index, num_classes, sparsity, explain_graph=False):
    node_masks = []
    if masks[0].shape[0] != edge_index.shape[1]:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    num_edges = edge_index.shape[1]
    chosen_nodes = num_nodes - int(num_nodes * sparsity)

    for pred in range(num_classes):
        mask = np.array(masks[pred].detach().cpu())
        args = np.argsort(-mask)
        if not explain_graph:
            node_set = {0}
        else:
            node_set = set()
        for index in args:
            src, dst = edge_index[0][index].item(), edge_index[1][index].item()
            node_set = node_set.union({src, dst})
            if len(node_set) >= chosen_nodes:
                break
        
        node_mask = torch.zeros(num_nodes)
        node_mask[list(node_set)] = 1
        node_masks.append(node_mask)
    
    return node_masks


def control_sparsity(masks, num_nodes, sparsity, num_classes, explain_graph=False):
    node_masks = []

    for pred in range(num_classes):
        node_mask = np.ones(num_nodes, dtype=np.int64)
        if not explain_graph:
            mask = np.array(masks[pred][1:])
            rnk = np.argsort(mask) + 1
        else:
            mask = np.array(masks[pred])
            rnk = np.argsort(mask)
        node_mask[rnk[:int(num_nodes * sparsity)]] = 0
        node_masks.append(node_mask)
    
    return node_masks


def eval_related_pred(model, x, edge_index, masks, num_nodes, num_classes):
    result = []

    model.eval()
    with torch.no_grad():
        def prob_with_mask(mask, pred):
            x_mask = x.clone().detach()
            x_mask[mask == 1, :] = 0
            return torch.softmax(model(x_mask, edge_index)[0], dim=0)[pred].item()
        
        for pred in range(num_classes):
            dict = {'zero': 0, 'mask': 0, 'maskout': 0, 'ori': 0}
            dict['ori'] = prob_with_mask(torch.zeros(num_nodes), pred)
            dict['maskout'] = prob_with_mask(masks[pred], pred)
            dict['mask'] = prob_with_mask(1 - masks[pred], pred)
            dict['zero'] = prob_with_mask(torch.ones(num_nodes), pred)
            result.append(dict)
        
        return result