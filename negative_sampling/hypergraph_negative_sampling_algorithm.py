import torch
import torch_geometric.nn.aggr as aggr
from enum import Enum
from torch import Tensor
from negative_sampling.hypergraph_negative_sampling import HypergraphNegativeSampler
from negative_sampling.hypergraph_negative_sampling_result import HypergraphNegativeSamplerResult, ABSizedHypergraphNegativeSamplerResult

class ABSizedHypergraphNegativeSampler(HypergraphNegativeSampler):
    class Mode(Enum):
        BEST_EFFORT = "best"
        NODE_AWARE = "node"
        HYPEREDGE_AWARE = "hyperedge"

    def __init__(self, num_node, alpha: float | int = 0.5, beta: int = 1, mode: Mode = Mode.BEST_EFFORT, avoid_duplicate_nodes: bool = True):
        super().__init__(num_node)
        self.alpha = alpha
        if self.alpha >= 1 and self.alpha != int(self.alpha):
            raise ValueError("If alpha is greater than or equal to 1, it must be an integer")
        self.beta = beta
        self.mode = mode
        self.avoide_duplicate_nodes = avoid_duplicate_nodes

    def fit(self, *args):
        return self
    
    #Parameter edge_index unused
    def get_probabilities(self, replace_mask: torch.Tensor, edge_index: torch.Tensor) -> float:
        probabilities = torch.ones((replace_mask.sum().item(), self.num_node), device = self.device)
        probabilities /= probabilities.sum(dim = 1, keepdim = True)
        return probabilities
    
    def get_replace_mask(self, edge_index: torch.Tensor) -> Tensor:
        if self.alpha >= 1:
            replace_mask = torch.zeros(edge_index.shape[1], dtype = torch.bool, device = self.device)
            """ The degrees return a tensor which for every position return the number of the verts
                presents in a hyperedge, like the file with the exention nverts.txt
            """
            degrees = aggr.SumAggregation()(
                torch.ones((edge_index.shape[1],1), dtype = torch.float32, device = self.device),
                edge_index[1]
            ).flatten().long()
            cursor = 0 
            for e in torch.unique(edge_index[1]):
                if degrees[e] <= self.alpha:
                    replace_mask[cursor:cursor + degrees[e]] = True
                else:
                    choise = torch.randint(0, degrees[e], (self.alpha,), device= self.device)
                    replace_mask[cursor + choise] = True
                cursor += degrees[e]
        else:
            replace_mask = torch.rand(edge_index.shape[1], device=self.device) >= self.alpha
            while True: #Ensure that at least one node is replaced in all hyperedges
                unchanged = (aggr.SumAggregation()(
                    replace_mask.float().view(-1,1),
                    edge_index[1]
                ) == 0).flatten().bool()
                if not unchanged[torch.unique(edge_index[1])].any():
                    break
                unchanged = torch.isin(edge_index[1], unchanged.nonzero())
                replace_mask[unchanged] = torch.rand(unchanged.sum().int().item(), device=self.device) >= self.alpha
        
        return replace_mask
    
    def generate(self, edge_index: Tensor) -> ABSizedHypergraphNegativeSamplerResult:
        positive_edge_index = edge_index[:, torch.argsort(edge_index[1])]
        negative_edge_index = torch.empty((2,0), dtype= torch.long, device= self.device)
        num_hyperedges = 0
        global_positives = torch.empty((2,0), dtype=torch.float32, device= self.device)
        global_replace_mask = torch.empty((0,), dtype=bool, device= self.device)
        global_replacement = torch.empty((0,) , dtype=torch.long, device=self.device)
    
        for _ in range(self.beta):
            local_edge_index = torch.clone(positive_edge_index)
            replace_mask = self.get_replace_mask(local_edge_index)
            probabilities = self.get_probabilities(replace_mask, positive_edge_index)
            _probabilities = torch.clone(probabilities).detach()
            if self.mode == self.Mode.BEST_EFFORT:
                pass
            elif self.mode == self.Mode.NODE_AWARE:
                nodes = local_edge_index[0, replace_mask]
                _probabilities[torch.arange(0, replace_mask.long().sum()), nodes] = 0
            elif self.mode == self.Mode.HYPEREDGE_AWARE:
                #Change the probabilities of the node in the hyperedge to 0
                for e in torch.unique(local_edge_index[1,replace_mask]):
                    edges = (local_edge_index[1, replace_mask] == 0).nonzero()
                    nodes = local_edge_index[0, local_edge_index[1] == e]
                    _probabilities[edges, nodes] = 0
            else:
                raise ValueError("Invalid mode")
            #Sampling
            _probabilities = _probabilities.sum(dim = 1, keepdim= True)
            #Avoid sampling duplicate nodes within the same hyperedge
            if self.avoide_duplicate_nodes: 
                replacement = torch.empty(replace_mask.sum().int().item(), dtype=torch.long, device= self.device)
                for e in torch.unique(local_edge_index[1, replace_mask]):
                    e_mask = local_edge_index[1, replace_mask] == e
                    replacement[e_mask] = torch.multinomial(_probabilities[e_mask], 1, replacement=False).flatten()
            else:
                replacement = torch.multinomial(_probabilities, 1, replacement=True).flatten()
            local_edge_index[0, replace_mask] = replacement
            local_edge_index[1] += num_hyperedges
            num_hyperedges = torch.max(local_edge_index[1]) + 1
            negative_edge_index = torch.cat([negative_edge_index , local_edge_index], dim = 1)
            global_positives = torch.cat([global_positives, probabilities], dim = 0)
            global_replace_mask = torch.cat([global_replace_mask, replace_mask], dim = 0)
            global_replacement = torch.cat([global_replacement, replacement], dim = 0) 

        return ABSizedHypergraphNegativeSamplerResult(
            global_positives,
            global_replace_mask,
            global_replacement,
            self,
            torch.clone(positive_edge_index),
            negative_edge_index
        )
    
class SizedHypergraphNegativeSampler(ABSizedHypergraphNegativeSampler):
    def __init__(self, num_node, *args, **kwargs):
        super(SizedHypergraphNegativeSampler, self).__init__(num_node, 0, 1,*args, **kwargs)

class MotifHypergraphNegativeSampler(HypergraphNegativeSampler):
    
    def fit(self, edge_index: Tensor, *args):
        return self
    
    def generate(self, edge_index: Tensor) -> HypergraphNegativeSampler:
        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=self.device),
            (self.num_node, edge_index.max().item() + 1),
            device= self.device
        )
        A = (sparse @ sparse.T).to_dense()
        A = A.where(A == 0, 1)#Clique expansion
        degrees = sparse.sum(dim=0).to_dense().flatten()
        edges = A.nonzero()
        generated_hyperedges_count = 0
        generated_hyperedges = []
        unique_hyperedges = torch.unique(edge_index[1])
        for i in range(unique_hyperedges.shape[0]):
            while True:
                degree = degrees[torch.randint(0, degrees.shape[0], (1,))].item()
                f = edges[torch.randint(0, edges.shape[0], (1,))].flatten()
                while f.shape[0] < degree:
                    probabilities = A[f].sum(dim = 0)
                    probabilities[f] = 0
                    probabilities = torch.where(probabilities == 1, 1., 0.)
                    if probabilities.sum() == 0:
                        break
                    probabilities /= probabilities.sum()
                    f = torch.cat([
                        f,
                        torch.multinomial(probabilities, 1)
                    ])
                if f.shape[0] < degree:
                    continue
                break
            generated_hyperedges.append(torch.vstack([
                f,
                torch.full((1, f.shape[0]),generated_hyperedges_count, device = self.device)
            ]))
            generated_hyperedges_count += 1
        
        return HypergraphNegativeSamplerResult(
            self,
            edge_index,
            torch.cat(generated_hyperedges, dim = 1)
        )
    
class CliqueHypergraphNegativeSampler(HypergraphNegativeSampler):

    def fit(self, edge_index: Tensor,*args):
        return self
    
    def generate(self, edge_index: torch.Tensor) -> HypergraphNegativeSamplerResult:
        sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device = self.device),
            (self.num_node, edge_index.max().item() + 1),
            device= self.device
        )

        A = (sparse @ sparse.T).to_dense()

        unique_edges = torch.unique(edge_index[1])
        generated_hyperedges_count = 0
        generated_hyperedges = []
        for _, in range(unique_edges.shape[0]):
            while True:
                #Randomly sample an hyperedge
                hyperedge = unique_edges[torch.randint(0, unique_edges.shape[0], (1,))]
                nodes = edge_index[0, edge_index[1] == hyperedge] #Get the nodes in the hyperedge
                #Randomly sample a node for removal
                hyperedge_mask = torch.zeros(nodes.shape[0], dtype = torch.bool, device = self.device)
                hyperedge_mask[0] = True #Randomly sample a node for removal
                p = aggr.MulAggregation()(
                    A[nodes[~hyperedge_mask]],
                    torch.zeros(hyperedge_mask.sum().item(), dtype = torch.int64,device = self.device)
                ).flatten()
                p[nodes] = 0
                p = p.where(p == 0, 1)
                if p.sum() == 0:
                    continue
                p /= p.sum()
                generated_hyperedges.append(torch.vstack([
                    torch.hstack([nodes[~hyperedge_mask], torch.multinomial(p,1)]),
                    torch.full((1, nodes.shape[0]),generated_hyperedges_count, device= self.device)
                ]))
                generated_hyperedges_count += 1
                break
        return HypergraphNegativeSamplerResult(
            self,
            edge_index,
            torch.cat(generated_hyperedges, dim=1)
        )