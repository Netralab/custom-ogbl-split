import torch

#from ogb.linkproppred import PygLinkPropPredDataset

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx




    
adj = nx.adjacency_matrix(nx.read_edgelist("Drug-Drug.edgelist", create_using=nx.DiGraph()))


split_edge = torch.load(path)

pos_train_edge = split_edge['train']#['edge']#.to(h.device)
pos_valid_edge = split_edge['valid']#['edge']#.to(h.device)
pos_test_edge = split_edge['test']#['edge_neg']#.to(h.device)
neg_valid_edge = split_edge['valid_neg']#['edge']#.to(h.device)
neg_test_edge = split_edge['test_neg']#['edge_neg']#.to(h.device)

	
row = pos_train_edge[0].numpy()
col = pos_train_edge[1].numpy()
edge_num = len(row)
data = np.ones( edge_num )
adj_train = csr_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[0]))
	
val_edges = pos_valid_edge.cpu().detach().numpy()
val_edges_false = neg_valid_edge.cpu().detach().numpy()
test_edges = pos_test_edge.cpu().detach().numpy()
test_edges_false = neg_test_edge.cpu().detach().numpy()
    
   
print(adj_train.count_nonzero())
print(val_edges.shape)
print(val_edges)
print(val_edges_false.shape)
print(val_edges_false)
print(test_edges.shape)
print(test_edges)
print(test_edges_false.shape)
print(test_edges_false) 





