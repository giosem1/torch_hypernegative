import torch
import pathlib
from torch.utils.data import Dataset
from torch import Tensor

class HypergraphBaseData(Dataset):
    
    def __getitem__(self, index: int) -> Tensor:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError

    def __init__(self, dataset_name: str, root: str = 'datasets'):
        super(HypergraphBaseData, self).__init__()
        self.dataset_name = dataset_name
        self.root_path = pathlib.Path(root)
        self.dataset_path = self.root_path / dataset_name
        
        self.edge_index = torch.load(open((self.dataset_path / "edge-index.pkl"), "rb"))
    
    def edge_index(self):
        return self.edge_index
    
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
    