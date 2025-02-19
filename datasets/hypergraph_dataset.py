import gdown
import tarfile
import pathlib
import torch
from data import HypergraphBaseData
from os import remove
from torch import Tensor

class ARBDataset(HypergraphBaseData): 
    """A object that save a hypergraph dataset from Google Drive and load in memory

        Args:
            dataset_name: which indicate the name of the dataset.
            root: which indicate the root where the dataset will be saved.
                 (default: 'datasets')
    """

    GDRIVE_IDs = {
        'coauth-DBLP': '15YpIK8vvzQJXyQC4bt-Sz951e4eb0rc_',
        "coauth-MAG-Geology": "14MOWsEJyNGiFKumvmgMcU1CqekDBayYV",
        "email-Enron": "1tTVZkdpgRW47WWmsrdUCukHz0x2M6N77",
        "tags-math-sx": "1eDevpF6EZs19rLouNpiKGLIlFOLUfKKG",
        "contact-high-school": "1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu",
        "contact-primary-school": "1sBHSEIyvVKavAho524Ro4cKL66W6rn-t",
        "NDC-substances": "1dLJt3qzAOYieay03Sp9h8ZfVMiU-nMqC"
    }
    

    def __init__(self, dataset_name: str, root: str = 'datasets'):
        self.dataset_name = dataset_name
        self.root_path = pathlib.Path(root)
        self.dataset_path = self.root_path / dataset_name
        if not self.dataset_path.exists():
            self.download()
        
        if not (self.dataset_path / "edge-index.pkl").exists():
            self.process()
        
        super(ARBDataset, self).__init__(dataset_name, root)

    def download(self) -> None:
        """ Take the dataset from Google Drive through the name of dataset,
            which is associeted the key, and download it in memory in the folder
            datasets.
        """

        if not self.dataset_path.exists():
            self.dataset_path.mkdir(parents=True)
        gdown.download(id = ARBDataset.GDRIVE_IDs[self.dataset_name], output = str(self.dataset_path / "raw.tar.gz"))
        file = tarfile.open(str(self.dataset_path / "raw.tar.gz"))
        file.extractall(self.dataset_path / "../")
        remove(str(self.dataset_path / "raw.tar.gz"))

    def process(self) -> None:
        """ Process the file in a tensor containing two list
            the first contain the nodes's id
            and the second the index of the nodes's numbers
            and then serialize the tensor 
        """
        list = [[],[]]

        with open(self.dataset_path / (self.dataset_name + "-nverts.txt") , "r") as f:
            with open(self.dataset_path / (self.dataset_name + "-simplices.txt"), "r") as g:
                for i, line in enumerate(f):
                    nverts = int(line) 
                    for _ in range(nverts):
                        simplices = int(g.readline()) - 1
                        list[0].append(simplices)
                        list[1].append(i)
       
        self.edge_index = torch.tensor(list)
        torch.save(self.edge_index, open((self.dataset_path / "edge-index.pkl"), "wb"))


class HypergraphDataset(HypergraphBaseData):
    """ A object that load from the memory the dataset's edge_index

        Args:
            dataset_name: which indicate the name of the dataset.
    """
    def __init__(self, dataset_name: str):
        ARBDataset(dataset_name)
        super(HypergraphDataset, self).__init__(dataset_name)

    def __getitem__(self, idx: int) -> Tensor:
        """ Take a index and find in the tensor the index of the node's number
            which is associeted at the list of nodes's id.

            Args: index: the index of the node in the tensor edge_index
            return: the tensor containing a list of nodes's id from the file which finish with '-simplices.txt'
                    and a list of the index of the node's number.
        """        
        return self.edge_index[:, self.edge_index[1] == idx]             

    def __len__(self) -> int:
        """Return the number of the node in the hypergraph
        """
        return len(torch.unique(self.edge_index[1]))
