import torch

class AuxDataset(torch.utils.data.Dataset):

    def __init__(self,tokenized_script,data):
        self.script = tokenized_script
        self.nudity = data['sex_level']
        self.violence = data['violence_level']
        self.profanity = data['profanity_level']
        self.substance = data['substance_level']
        self.frightening = data['frightening_level']
        
    def __getitem__(self,idx):
        
        text = torch.tensor(self.script[idx])
        aspects = [self.nudity[idx],self.violence[idx],self.profanity[idx],self.substance[idx],self.frightening[idx]]
        label = torch.tensor(aspects)
        return text,label

    def __len__(self):
        return len(self.script)

