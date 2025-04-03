import torch
import torch_geometric.nn.aggr as aggr
from negative_sampling.hypergraph_negative_sampling import HypergraphNegativeSampler
from abc import ABC
from torch import Tensor

class HypergraphNegativeSamplerResult(ABC):

    def __init__(self, sampler: HypergraphNegativeSampler, p_edge_index: Tensor, n_edge_index: Tensor, ):
        self.sampler = sampler
        self.__p_edge_index = p_edge_index
        _, self.__p_edge_index[1] = torch.unique(self.__p_edge_index[1], return_inverse = True) 
        self.__n_edge_index = n_edge_index
        _, self.__n_edge_index[1] = torch.unique(self.__n_edge_index[1], return_inverse = True)         
        self.device = sampler.device

    def remove_positive_from_negative(self):
        #Negative sparse tensor
        n_sparse = torch.sparse_coo_tensor(
            self.__n_edge_index,
            torch.ones_like(self.__n_edge_index[1], dtype = torch.float32, device = self.device),
            (self.sampler.num_node, self.num_n_edges),
            device = self.device
        ).coalesce()
        #Positive sparse tensor
        p_sparse = torch.sparse_coo_tensor(
            self.__p_edge_index,
            torch.ones_like(self.__p_edge_index[1], dtype = torch.float32, device = self.device),
            (self.sampler.num_node, self.num_p_edges),
            device = self.device
        ).coalesce()

        #Calcuate the matrix between the negative's nodes and positive's nodes
        A = n_sparse.T @ p_sparse
        n_degrees = torch.sparse.sum(n_sparse, dim = 0).to_dense()
        p_degrees = torch.sparse.sum(p_sparse, dim = 0).to_dense()
        #Check if in the degree of the negative samples are in the positive samples
        dmask = n_degrees[A.indices()[0] == p_degrees[A.indices()[1]]]
        #Check if in the matrix are presents the degree of negative samples
        vmask = A.values() == n_degrees[A.indices()[0]]
        mask = ~aggr.SumAggregation()(
            (vmask & dmask).view(-1,1),
            A.indices()[0]
        ).flatten()

        self.num_p_in_n = self.num_n_edges - mask.sum()
        self.n_in_p_mask = torch.isin(self.__n_edge_index[1], mask.nonzero().flatten())
        self.__n_edge_index = self.__n_edge_index[:,self.n_in_p_mask]
        _, self.__n_edge_index[1] = torch.unique(self.__n_edge_index[1], return_inverse = True)
        return self
    
    def oversample(self):
        num_samples = self.num_p_edges - self.num_n_edges
        if num_samples <= 0:
            self.mask = torch.zeros_like(self.__n_edge_index[1], dtype = torch.bool, device= self.device)
            return self
        unique_n_indices = torch.unique(self.__n_edge_index[1])
        #Generate a boolean mask of num_samples
        e_mask = torch.zeros_like(unique_n_indices, dtype= torch.bool)
        e_mask[:num_samples] = True
        #Mask of duplicate edges
        e_mask = e_mask[torch.randperm(e_mask.shape[0])]
        self.mask = torch.isin(self.__n_edge_index[1], unique_n_indices[e_mask])
        #Create a temporal negative edge index
        temp_n_edge_index = self.__n_edge_index[:, self.mask]
        _, temp_n_edge_index[1] = torch.unique(temp_n_edge_index[1], return_inverse = True)
        temp_n_edge_index[1] += self.num_n_edges
        
        self.__n_edge_index = torch.hstack([self.__n_edge_index, temp_n_edge_index])
        
        return self

    def clean(self):
        self.remove_positive_from_negative()
        self.oversample()
        return self

    @property
    def num_p_edges(self):
        return torch.unique(self.__p_edge_index[1]).shape[0]
    
    @property
    def num_n_edges(self):
        return torch.unique(self.__n_edge_index[1]).shape[0]
    
    @property
    def num_edges(self):
        return torch.unique(self.edge_index[1]).shape[0]
    
    @property
    def edge_index(self) -> Tensor:
        max_index = torch.max(self.p_edge_index[1]) + 1
        n_edge_index = torch.clone(self.n_edge_index)
        n_edge_index[1] += max_index

        return torch.hstack([self.p_edge_index, n_edge_index])

    @property
    def y(self) -> Tensor:
        return torch.vstack([self.y_p, self.y_n])
    
    @property
    def y_p(self) -> Tensor:
        return torch.ones((torch.unique(self.__p_edge_index[1]).shape[0], 1), device = self.device)
    
    @property
    def y_n(self) -> Tensor:
        return torch.ones((torch.unique(self.__n_edge_index[1]).shape[0], 1), device = self.device)
    
    @property
    def negative_mask(self):
        return ~self.y.type(torch.bool).flatten()
    
    @property
    def positve_mask(self):
        return self.y.type(torch.bool).flatten()
    
    def __repr__(self):
        return self.edge_index.__repr__()

    @property
    def p_edge_index(self) -> Tensor:
        return self.__p_edge_index
    
    @property
    def n_edge_index(self) -> Tensor:
        return self.__n_edge_index
    
class ABSizedHypergraphNegativeSamplerResult(HypergraphNegativeSamplerResult):

    def __init__(self, p: torch.Tensor, replace_mask: torch.Tensor, replacement: torch.Tensor, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.p = p
        self.replace_mask = replace_mask
        self.replacement = replacement

    def remove_positive_from_negative(self):
        super().remove_positive_from_negative()
        self.replacement = self.replacement[(self.replace_mask & self.n_in_p_mask)[self.replace_mask]]
        self.p = self.p[(self.replace_mask & self.mask)[self.replace_mask]]
        self.replace_mask = self.replace_mask[self.n_in_p_mask]
        
        return self
    
    def oversample(self):
        super().oversample()
        temp_replacement = self.replacement[(self.replace_mask & self.mask)[self.replace_mask]]
        temp_p = self.p[(self.replace_mask & self.mask)[self.replace_mask]]
        temp_replace_mask = self.replace_mask[self.mask]
        self.replacement = torch.hstack([self.replacement, temp_replacement])
        self.p = torch.vstack([self.p, temp_p])
        self.replace_mask = torch.hstack([self.replace_mask, temp_replace_mask])

        return self