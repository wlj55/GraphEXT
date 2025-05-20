import os
import os.path as osp
import torch
from model.models import *
from dataset.data import *
import argparse
from torch.nn.functional import cross_entropy
from torch_geometric.loader import DataLoader


def accuracy(pred, y):
    return (pred == y).sum() / y.shape[0]


def accuracy_dataloader(device, model, dataloader):
    pred, y = [], []
    for data in dataloader:
        data = data.to(device)
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        pred.append(model(x, edge_index, batch).argmax(dim=1))
        y.append(data.y.view(-1))
    
    pred = torch.cat(pred, dim=0)
    y = torch.cat(y, dim=0)
    return (pred == y).sum() / y.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Graph-Twitter', 
                        choices=['BA_shapes', 'BA_LRP', 'BBBP', 'ClinTox', 'Graph-SST2', 'Graph-Twitter'])
    parser.add_argument('--model_used', type=str, default='GCN_3l', 
                        choices=['GCN_2l', 'GCN_3l', 'GIN_2l', 'GIN_3l'])
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    data_path = './dataset'
    checkpoint_path = osp.join('model', 'checkpoint', args.dataset)
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model_save_path = osp.join(checkpoint_path, args.model_used + '.pkl')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data, num_nodes, dim_node, num_classes = load_dataset(data_path, args.dataset)

    if args.dataset in ['BA_shapes']:
        model_level = 'node'
        model = eval(args.model_used)(model_level=model_level, dim_node=dim_node,
                                dim_hidden=args.dim_hidden, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-4)
        data = data.to(device)

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = cross_entropy(pred[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                pred = model(data.x, data.edge_index).argmax(dim=1)
                print(f'epoch #{epoch:3d}, loss = {loss:.4f}, ', 
                    f'train_acc = {accuracy(pred[data.train_mask], data.y[data.train_mask]):.4f}, ',
                    f'valid_acc = {accuracy(pred[data.val_mask], data.y[data.val_mask]):.4f}, ',
                    f'test_acc = {accuracy(pred[data.test_mask], data.y[data.test_mask]):.4f}',)
        
        torch.save(model.state_dict(), model_save_path)
        
    else:
        model_level = 'graph'
        model = eval(args.model_used)(model_level=model_level, dim_node=dim_node,
                            dim_hidden=args.dim_hidden, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-4)

        train_loader = DataLoader(data['train'], batch_size=1, shuffle=True)
        valid_loader = DataLoader(data['val'], batch_size=1, shuffle=True)
        test_loader = DataLoader(data['test'], batch_size=1, shuffle=True)

        sum = 0
        for data in train_loader:
            sum += data.num_nodes
        for data in valid_loader:
            sum += data.num_nodes
        for data in test_loader:
            sum += data.num_nodes
        print(len(train_loader) + len(valid_loader) + len(test_loader), sum)
        exit(0)

        best = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                x = data.x
                edge_index = data.edge_index
                batch = data.batch

                logits = model(x, edge_index, batch)
                loss = cross_entropy(logits, data.y.view(-1)).to(device)
                loss.backward()
                optimizer.step()

                total_loss += loss * data.num_graphs
            
            model.eval()
            acc_test = accuracy_dataloader(device, model, test_loader)
            print(f'epoch #{epoch:3d}, loss = {total_loss / len(train_loader):.4f}, ', 
                f'train_acc = {accuracy_dataloader(device, model, train_loader):.4f}, ',
                f'valid_acc = {accuracy_dataloader(device, model, valid_loader):.4f}, ',
                f'test_acc = {acc_test:.4f}',)
            
            if epoch > args.epochs // 2 and acc_test >= best:
                best = acc_test
                torch.save(model.state_dict(), model_save_path)
            
        print('best test acc:', best)


if __name__ == '__main__':
    main()