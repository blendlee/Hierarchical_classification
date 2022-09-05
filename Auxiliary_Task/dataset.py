import torch

class AuxDataset(torch.utils.data.Dataset):

    def __init__(self,tokenized_abstract,data):
        self.abstract = tokenized_abstract
        self.y= data['Y']
        self.num_categories = len(self.y.unique())

        
    def __getitem__(self,idx):
        
        text = torch.tensor(self.script[idx])
        categories = [0]*self.num_categories
        categories[self.y[idx]]=1
        label = torch.tensor(categories)
        return text,label

    def __len__(self):
        return len(self.abstract)

