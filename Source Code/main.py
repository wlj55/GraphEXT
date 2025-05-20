import argparse
import os
import hydra
import os.path as osp
import scipy.sparse as ssp
import time
import torch
import torch_geometric
from model.models import *
from dataset.data import *
from method.flowx import FlowX
from method.graphext import GraphEXT
from method.gnnexplainer import GNNExplainer
from method.gradcam import GradCAM
from method.pgexplainer import PGExplainer
from method.evaluation import *
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader

from dig.xgraph.utils.compatibility import compatible_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BBBP', 
                        choices=['BA_shapes', 'BA_LRP', 'BBBP', 'ClinTox', 'Graph-SST2', 'Graph-Twitter'])
    parser.add_argument('--model_used', type=str, default='GIN_3l', 
                        choices=['GCN_2l', 'GCN_3l', 'GIN_2l', 'GIN_3l'])
    parser.add_argument('--explainer', type=str, default='GradCAM',
                        choices=['FlowX', 'GNNExplainer', 'GradCAM', 'GraphEXT', 'PGExplainer'])
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--dim_hidden', type=int, default=300)
    args = parser.parse_args()

    data_path = './dataset'
    log_path = osp.join('log', args.dataset, args.explainer, args.model_used)
    log_file = log_path + '/Sparsity=' + str(args.sparsity) + '.log'
    if not osp.exists(log_path):
        os.makedirs(log_path)
    checkpoint_path = './model/checkpoint'
    model_save_path = osp.join(checkpoint_path, args.dataset, args.model_used + '.pkl')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fid, fid_inv = 0, 0
    
    with open(log_file, 'a') as f:
        print('model used:', args.model_used, file=f)
        print('method used:', args.explainer, file=f)
        
    data, num_nodes, dim_node, num_classes = load_dataset(data_path, args.dataset)
    if args.dataset in ['BA_shapes']:
        model = eval(args.model_used)(model_level='node', dim_node=dim_node,
                                    dim_hidden=args.dim_hidden, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_save_path))

        if args.explainer == 'PGExplainer':
            explainer = eval(args.explainer)(model, in_channels=900, device=device, explain_graph=False)
            tmp_file = osp.join(checkpoint_path, args.dataset, args.model_used + 'PGExplainer.pt')
            if not osp.exists(tmp_file):
                explainer.train_explanation_network([data])
                torch.save(explainer.state_dict(), tmp_file)
            state_dict = torch.load(tmp_file)
            explainer.load_state_dict(state_dict)
        else:
            explainer = eval(args.explainer)(model, explain_graph=False)

        data = data.cpu()
        node_indices = torch.where(data.test_mask * data.y != 0)[0].tolist()
        label = data.y[node_indices]
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
                    (edge_weight, (data.edge_index[0], data.edge_index[1])),
                    shape=(data.num_nodes, data.num_nodes)
                )
        data_list = extract_enclosing_subgraphs(node_indices, A, data.x, label, int(args.model_used[4]))

        for index, data in enumerate(data_list):
            print(f'explain node #{node_indices[index]:d}')
            data.to(device)
            prediction = model(data.x, data.edge_index)[0].argmax(-1).item()
            print(prediction, data.y)

            print(data)
            _masks = explainer(data.x, data.edge_index, sparsity=1, num_classes=num_classes, node_idx=0, max_nodes=int(data.num_nodes * (1 - args.sparsity)))
            
            masks = _masks.copy()
            log_file = log_path + '/Sparsity=' + str(args.sparsity) + '.log'
            masks = convert_edge_mask(masks, data.num_nodes, data.edge_index, num_classes, args.sparsity)
            masks = control_sparsity(masks, data.num_nodes, args.sparsity, num_classes)
            print(masks[prediction])
            result = eval_related_pred(model, data.x, data.edge_index, masks, data.num_nodes, num_classes)
            print(result[prediction])
            print(f"Fidelity = {result[prediction]['ori'] - result[prediction]['maskout']:.4f}\n", 
                f"Fidelity_inv = {result[prediction]['ori'] - result[prediction]['mask']:.4f}\n",
                f"Sparsity = {args.sparsity:.4f}")
            with open(log_file, 'a') as f:
                print(f'explain node #{node_indices[index]:d},', end=' ', file=f)
                print(f"({result[prediction]['ori'] - result[prediction]['maskout']:.4f}, "
                    f"{result[prediction]['ori'] - result[prediction]['mask']:.4f})\n", file=f)
            fid += result[prediction]['ori'] - result[prediction]['maskout']
            fid_inv += result[prediction]['ori'] - result[prediction]['mask']

        log_file = log_path + '/Sparsity=' + str(args.sparsity) + '.log'
        with open(log_file, 'a') as f:
            print(f"Final:\n Fidelity = {fid / len(node_indices):.4f}\n", 
                f"Fidelity_inv = {fid_inv / len(node_indices):.4f}\n",
                f"Sparsity = {args.sparsity:.4f}\n", file=f)
    else:
        model = eval(args.model_used)(model_level='graph', dim_node=dim_node,
                                    dim_hidden=args.dim_hidden, num_classes=num_classes).to(device)
        
        model.load_state_dict(torch.load(model_save_path))

        if args.explainer == 'PGExplainer':
            explainer = eval(args.explainer)(model, in_channels=600, device=device, explain_graph=True)
            tmp_file = osp.join(checkpoint_path, args.dataset, args.model_used + 'PGExplainer.pt')
            if not osp.exists(tmp_file):
                explainer.train_explanation_network(data['train'])
                torch.save(explainer.state_dict(), tmp_file)
            state_dict = torch.load(tmp_file)
            explainer.load_state_dict(state_dict)
        else:
            explainer = eval(args.explainer)(model, explain_graph=True)

        data_loader = DataLoader(data['test'], batch_size=1, shuffle=False)
        for index, data in enumerate(data_loader):
            if data.num_nodes == 1:
                continue
            print(f'explain graph #{index + 1:d}')
            data.to(device)
            prediction = model(data.x, data.edge_index)[0].argmax(-1).item()
            masks = explainer(data.x, data.edge_index, sparsity=0, num_classes=num_classes, node_idx=0, max_nodes=int(data.num_nodes * (1 - args.sparsity)))
            
            log_file = log_path + '/Sparsity=' + str(args.sparsity) + '.log'
            masks = convert_edge_mask(masks, data.num_nodes, data.edge_index, num_classes, args.sparsity, explain_graph=True)
            masks = control_sparsity(masks, data.num_nodes, args.sparsity, num_classes, explain_graph=True)
            print(masks[prediction])
            result = eval_related_pred(model, data.x, data.edge_index, masks, data.num_nodes, num_classes)
            print(result[prediction])
            print(f"Fidelity = {result[prediction]['ori'] - result[prediction]['maskout']:.4f}\n", 
                f"Fidelity_inv = {result[prediction]['ori'] - result[prediction]['mask']:.4f}\n",
                f"Sparsity = {args.sparsity:.4f}")
            with open(log_file, 'a') as f:
                print(f'explain graph #{index + 1:d}', end=' ', file=f)
                print(f"({result[prediction]['ori'] - result[prediction]['maskout']:.4f}, "
                    f"{result[prediction]['ori'] - result[prediction]['mask']:.4f})\n", file=f)
            fid += result[prediction]['ori'] - result[prediction]['maskout']
            fid_inv += result[prediction]['ori'] - result[prediction]['mask']
            
        log_file = log_path + '/Sparsity=' + str(args.sparsity) + '.log'
        with open(log_file, 'a') as f:
            print(f"Final:\n Fidelity = {fid / len(data_loader):.4f}\n", 
                f"Fidelity_inv = {fid_inv / len(data_loader):.4f}\n",
                f"Sparsity = {args.sparsity:.4f}\n", file=f)


if __name__ == '__main__':
    main()