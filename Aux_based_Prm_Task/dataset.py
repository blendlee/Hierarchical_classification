import torch

class AuxBasedPrmDataset(torch.utils.data.Dataset):

    def __init__(self,tokenized_abstract,att,data):
        self.abstract = tokenized_abstract
        self.word_attention = att
        self.domain = data['Y1']
        
    def __getitem__(self,idx):
        
        text = torch.tensor(self.abstract[idx])
        attention_label = torch.tensor(self.word_attention[idx])
        label = torch.tensor(int(self.domain[idx]))
        return text,attention_label,label

    def __len__(self):
        return len(self.abstract)

