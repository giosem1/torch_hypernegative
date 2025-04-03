import torch
import pathlib
from abc import ABC
from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple

class HypergraphBaseData(ABC, Dataset):
    """ A class which obtains the the edge_index and time_saved from a dataset

        Args:
            dataset_name (string): The dataset's name.
            edge_index (Tensor): A tensor where are saved the node's id.
            time_saved (Tensor): A tensor where are saved the timestamp.
            nvert_attribute (Tensor): A tensor where are saved the node's attribute.
            root (string, optional): The folder's name where the dataset will be saved.
                 (default: 'datasets')
    """

    def __init__(self, dataset_name: str, edge_index: Tensor, time_saved: Tensor, nverts_attribute: Tensor, root: str = 'datasets' ):
        super(HypergraphBaseData, self).__init__()
        self.dataset_name = dataset_name
        self.root_path = pathlib.Path(root)
        self.dataset_path = self.root_path / dataset_name
        self.__edge_index = edge_index
        self.__time_saved = time_saved
        self.__nverts_attribute = nverts_attribute
        
    @property
    def edge_index(self) -> Tensor:
        return self.__edge_index
    
    @property
    def time_saved(self) -> Tensor:
        return self.__time_saved
    
    @property
    def nverts_saved(self) -> Tensor:
        return self.__nverts_attribute
    
    def __getitem__(self, idx: int) -> Tuple:
        """ Take a index and find in the tensor the index of the node's number
            which is associeted at the list of nodes's id.

            Args: 
                index (int): the index of the node in the tensor edge_index
                return: the tuple containing as first element a tensor which contains a list of nodes's id 
                    from the file which finish with '-simplices.txt'
                    and a list of the index of the node's number,
                    and as second element the time which is contained in the list of the time_stampded.
        """        
        
        return self.edge_index[:, self.edge_index[1] == idx].clone(), self.__time_saved[1,self.__time_saved[0] == idx].clone()
    
    def __len__(self) -> int:
        """Return the number of the node in the hypergraph
        """
        return len(torch.unique(self.__edge_index[1]))

    
    def generate_timestamped(self) -> None:
        """ Process the file of the timestamp and generate a tensor which contains two lists 
            the first contains the position in the of the the timestamp 
            the second contains the value of the timestamp
        """
        timestamped_list = [[],[]]
        with open(self.dataset_path / (self.dataset_name + "-times.txt"), "r") as f:
            for i, line in enumerate(f):
                timestamped = int(line)
                timestamped_list[0].append(i)
                timestamped_list[1].append(timestamped)

        self.__time_saved = torch.tensor(timestamped_list)
        torch.save(self.__time_saved, open((self.dataset_path / "times-index.pkl"), "wb")) 