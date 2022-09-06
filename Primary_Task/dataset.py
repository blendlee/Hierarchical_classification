import torch

class PrmDataset(torch.utils.data.Dataset):

    def __init__(self,tokenized_abstract,data):
        self.abstract = tokenized_abstract
        self.domain = data['Y1']
        
    def __getitem__(self,idx):
        
        text = torch.tensor(self.abstract[idx])
        label = torch.tensor(int(self.domain[idx]))
        return text,label

    def __len__(self):
        return len(self.abstract)

