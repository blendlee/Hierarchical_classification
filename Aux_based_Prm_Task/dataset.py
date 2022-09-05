import torch

class AuxBasedPrmDataset(torch.utils.data.Dataset):

    def __init__(self,tokenized_abstract,data):
        self.abstract = tokenized_abstract
        self.domain = data['Y1']
        self.area = data['Y']
        
    def __getitem__(self,idx):
        
        text = torch.tensor(self.abstract[idx])
        area = torch.tensor(self.area[idx])
        label = torch.tensor(self.domain[idx])
        return text,area,label

    def __len__(self):
        return len(self.abstract)

