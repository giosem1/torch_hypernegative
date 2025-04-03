import torch
from typing import List,Any
from torch.utils.data import DataLoader
from hyperlink_prediction.datasets import HypergraphBaseData, ARBDataset

class DatasetLoader(DataLoader):
    """ A class data loader which merge data object from a dataset 
        to a mini-batch.
        
        Args:
            dataset (HypergraphBaseData): The dataset from which to load the data.
            batch_size (int, optional): How many samples per batch to load.
                (default: 1)
            shuffle (bool, optional): Set True to have data reshuffled at every epoch.
                (default: False)
            **kwargs: Additional arguments for the class.
    """

    def __init__(self, dataset: HypergraphBaseData, batch_size: int = 1, shuffle: bool = False, **kwargs):
        kwargs.pop("collate_fn", None)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn= self.collate,
            **kwargs
        )
    
    def collate(self, batch: List[Any]):
        return torch.hstack([b[0] for b in batch]), torch.vstack([v[1] for v in batch])