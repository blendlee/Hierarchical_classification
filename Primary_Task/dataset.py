import torch

class PrmDataset(torch.utils.data.Dataset):

    def __init__(self,tokenized_script,data):
        self.script = tokenized_script
        self.rating = data['rating']
        
    def __getitem__(self,idx):
        
        text = torch.tensor(self.script[idx])
        rating = torch.tensor(self.rating[idx])
        return text,rating

    def __len__(self):
        return len(self.script)

