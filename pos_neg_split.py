# vessap open graph benchmark dataset

import os
import os.path as osp
import sys

from numpy.random import seed
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))

import math
import torch
import numpy as np
import pandas as pd
import scipy
import argparse
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx, negative_sampling
import random



from ogb.io import DatasetSaver


parser = argparse.ArgumentParser(description='generate OGB Link Prediction Dataset')
parser.add_argument('--dataset', help='Dataset name (without ogbl-).', type=str,required=True)
parser.add_argument('--splitting_strategy', help='Splitting Strategy: random or spatial.', type=str,required=True)
parser.add_argument('--number_of_workers', type=str, default=4)
parser.add_argument('--seed', type=int, default=123, help="Set the seed for torch, numpy and random functions.")
parser.add_argument('--data_root_dir', type=str, default='data')

args = parser.parse_args()

dataset_name = 'ogbl-' + args.dataset 



saver = DatasetSaver(dataset_name = dataset_name,
                    is_hetero = False,
                    version = 1)


# load PyTorch Geometrics Graph

dataset = from_networkx(nx.read_edgelist(args.dataset, create_using=nx.DiGraph(), nodetype=int))

num_data = 1 
graph_list = []
for i in range(1):
    #data = dataset[i]
    graph = dict()
    graph['num_nodes'] = int(dataset.num_nodes)
    #graph['node_feat'] = np.array(data.x)
    graph['edge_index'] = dataset.edge_index.numpy() # only train pos edge index, but both directions / undirected!
    #if use_edge_attr:
    #    graph['edge_feat'] = data.edge_attr.numpy()
    graph_list.append(graph)

print(graph_list)
saver.save_graph_list(graph_list)


split_edge = {'train': {}, 'valid': {}, 'test': {}}





num_edges_total = dataset.num_edges
train_ratio = 0.85
val_ratio = 0.05
test_ratio = 0.1

all_edges = np.array(list(dataset.edge_index))

# define indices of edges for train, val, test split
all_edge_indices = [x for x in range(0, num_edges_total)]
random.shuffle(all_edge_indices)
train_indices = np.array(all_edge_indices[:int(train_ratio*num_edges_total)])
val_indices = np.array(all_edge_indices[int(train_ratio*num_edges_total):int((train_ratio+val_ratio)*num_edges_total)])
test_indices = np.array(all_edge_indices[int((train_ratio+val_ratio)*num_edges_total):])


neg_samples = negative_sampling(dataset.edge_index, num_nodes= dataset.num_nodes,
                       num_neg_samples= math.floor(len(dataset.edge_index[0]) * 0.15), method='dense')


div = math.ceil(len(dataset.edge_index[0]) * val_ratio)




val_neg_row = neg_samples[0][:div]
val_neg_col = neg_samples[1][:div]
test_neg_row = neg_samples[0][div: ]
test_neg_col = neg_samples[1][div: ]





row_train = []
col_train = []

row_val = []
col_val = []

row_test = []
col_test = []

for i in range(train_indices.shape[0]):
    row_train.append(dataset.edge_index[0][train_indices[i]].item())
    col_train.append(dataset.edge_index[1][train_indices[i]].item())

for i in range(val_indices.shape[0]):
    row_val.append(dataset.edge_index[0][val_indices[i]].item())
    col_val.append(dataset.edge_index[1][val_indices[i]].item())

for i in range(test_indices.shape[0]):
    row_test.append(dataset.edge_index[0][test_indices[i]].item())
    col_test.append(dataset.edge_index[1][test_indices[i]].item())



train_edge_index = torch.stack((torch.tensor(row_train), torch.tensor(col_train)), dim=0)
val_edge_index = torch.stack((torch.tensor(row_val), torch.tensor(col_val)), dim=0)
test_edge_index = torch.stack((torch.tensor(row_test), torch.tensor(col_test)), dim=0)
val_neg_index = torch.stack((val_neg_row, val_neg_col), dim=0)
test_neg_index = torch.stack((test_neg_row, test_neg_col), dim=0)




# Save split indices 
split_edge = dict()
split_edge['train'] = train_edge_index  #train_indices 
split_edge['valid'] = val_edge_index    #val_indices
split_edge['test'] = test_edge_index    #test_indices
split_edge['valid_neg'] = val_neg_index
split_edge['test_neg'] = test_neg_index
saver.save_split(split_edge, split_name = 'random')


print(f'Number of nodes: {dataset.num_nodes}')
print(f'Number of edges: {dataset.num_edges}')




mapping_path = 'mapping/'

os.makedirs(mapping_path,exist_ok=True)
try:
    os.mknod(os.path.join(mapping_path, 'README.md'))
except:
    print("Readme.md already exists.")
saver.copy_mapping_dir(mapping_path)


saver.save_task_info(task_type = 'link prediction', eval_metric = 'acc')


meta_dict = saver.get_meta_dict()
print(meta_dict)


saver.zip()
saver.cleanup()





