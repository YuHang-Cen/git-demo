from typing import List,Tuple,Dict

import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torch_geometric.loader import NeighborLoader

def ReadInDisease2Target(path: str) -> Tensor:
    '''
    Output.shape = N,(pos_num + neg_num),3
    , where neg_num = 0 in train
    and 3 means (disease,target_gene,y)
    '''
    with open(path, 'r') as file:
        content = file.readlines()
        data = []
        col_num = len(content[0].strip().split(','))
        for line in content:
            triplets = line.strip().split(',')
            # Group every three elements into a triplet
            for i in range(0, col_num, 3):
                disease = int(triplets[i])
                target_gene = int(triplets[i + 1])
                y = int(triplets[i + 2])
                data.append([disease, target_gene, y])
        
    return torch.tensor(data, dtype=torch.long).reshape(-1,col_num//3,3)

def ShuffleDisease2Target(d2t: Tensor) -> Tensor:
    '''
    Shuffles the input tensor along the first dimension.
    '''
    indices = torch.randperm(d2t.size(0))
    return d2t[indices]


class HybridDataloader:
    def __init__(self,
                 g_path: str,
                 t_path: str,
                 batch_size: int,
                 num_neighbors: int,
                 num_neighbors_layers: int,
                 num_workers: int,
                 shuffle: bool = True,
                 drop_last: bool = True):
        
        graph = torch.load(g_path)
        d2t = ReadInDisease2Target(t_path)

        if shuffle:
            d2t = ShuffleDisease2Target(d2t)

        if drop_last:
            num_batches, num_res = divmod(d2t.size(0),batch_size)
            if num_res > 0:
                d2t = d2t[:num_batches * batch_size]

        diseases = d2t[:,:,0]
        genes = d2t[:,:,1]
        ys = d2t[:,:,2]
        triplet_num = d2t.size(1)

        input_nodes = {'disease':diseases.flatten(),'gene':genes.flatten()}
        self.graph_loaders = {
            node_type: NeighborLoader(
                graph,
                num_neighbors = [num_neighbors] * num_neighbors_layers,
                input_nodes=(node_type, node_ids),
                batch_size = batch_size * triplet_num,
                shuffle=False,
                num_workers = num_workers
            )
            for node_type, node_ids in input_nodes.items()
        }

        self.y_loader = DataLoader(TensorDataset(ys),batch_size=batch_size,num_workers = num_workers,shuffle=False)

    def __iter__(self):
        return zip(self.graph_loaders['disease'],self.graph_loaders['gene'], self.y_loader)
    
    def __len__(self):
        return len(self.y_loader)
    
class DualHybridDataloader(HybridDataloader):
    def __init__(self,
                 g_path: str,
                 t_path: str,
                 batch_size: int,
                 small_num_neighbors: int,
                 big_num_neighbors: int,
                 num_neighbors_layers: int,
                 num_workers: int,
                 shuffle: bool = True,
                 drop_last: bool = True):
        
        graph = torch.load(g_path)
        d2t = ReadInDisease2Target(t_path)

        if shuffle:
            d2t = ShuffleDisease2Target(d2t)

        if drop_last:
            num_batches, num_res = divmod(d2t.size(0),batch_size)
            if num_res > 0:
                d2t = d2t[:num_batches * batch_size]

        diseases = d2t[:,:,0]
        genes = d2t[:,:,1]
        ys = d2t[:,:,2]
        triplet_num = d2t.size(1)

        input_nodes = {'disease':diseases.flatten(),'gene':genes.flatten()}
        neighbor_nums_dict = {'big':big_num_neighbors,'small':small_num_neighbors}
        self.graph_loaders = {
            neighbor_num_type: {
                node_type: NeighborLoader(
                    graph,
                    num_neighbors = [num_neighbors] * num_neighbors_layers,
                    input_nodes=(node_type, node_ids),
                    batch_size = batch_size * triplet_num,
                    shuffle=False,
                    num_workers = num_workers
                )
                for node_type, node_ids in input_nodes.items()
            }
            for neighbor_num_type,num_neighbors in neighbor_nums_dict.items()
        }

        self.y_loader = DataLoader(TensorDataset(ys),batch_size=batch_size,num_workers = num_workers,shuffle=False)

    def __iter__(self):
        return zip(self.graph_loaders['small']['disease'],self.graph_loaders['small']['gene'],
                   self.graph_loaders['big']['disease'],self.graph_loaders['big']['gene'],self.y_loader)
    
    def __len__(self):
        return len(self.y_loader)