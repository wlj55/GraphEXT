from dig.xgraph.dataset import *
import torch
from torch.utils.data import random_split
from tqdm import tqdm
from torch_geometric.data import Data
import scipy.sparse as ssp
import random


def split_dataset(dataset, dataset_split=[0.8, 0.1, 0.1]):
    dataset_len = len(dataset)
    dataset_split = [int(dataset_len * dataset_split[0]),
                     int(dataset_len * dataset_split[1]),
                     0]
    dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
    train_set, val_set, test_set = random_split(dataset, dataset_split)

    return {'train': train_set, 'val': val_set, 'test': test_set}


def load_dataset(data_path, dataset):
    if dataset in ['BA_shapes']:
        dataset = SynGraphDataset(data_path, dataset)
        data = dataset[0]
        dim_node = dataset.num_node_features
        dim_edge = dataset.num_edge_features
        num_classes = dataset.num_classes
        return data, 1, dim_node, num_classes

    if dataset in ['Graph-SST2', 'Graph-Twitter']:
        dataset = SentiGraphDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y
    if dataset in ['BBBP', 'ClinTox']:
        dataset = MoleculeDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y[:, 0]
    if dataset in ['BA_2Motifs']:
        dataset = SynGraphDataset(data_path, dataset)
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y

    dataset.data.y = dataset.data.y.long()
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes

    splitted_dataset = split_dataset(dataset)
    return splitted_dataset, 1, dim_node, num_classes


def construct_pyg_graph(node_ids, adj, node_features, y):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])

    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, node_id=node_ids, num_nodes=num_nodes)
    return data


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A,
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(u, num_hops, A, node_features, y):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [u]
    dists = [0, 0]
    visited = set([u])
    fringe = set([u])
    for dist in range(1, num_hops + 1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)

        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, node_features, y


def extract_enclosing_subgraphs(node_indices, A, x, label, num_hops):
    data_list = []
    for idx, u in enumerate(tqdm(node_indices)):
        tmp = k_hop_subgraph(u, num_hops, A, node_features=x, y=label[idx])
        data = construct_pyg_graph(*tmp)
        data_list.append(data)

    return data_list