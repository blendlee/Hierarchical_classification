import argparse
from preprocess import  preprocess, process_data,split_data
from tokenizer import build_vocab,tokenize
from dataset import PrmDataset
from model import PrmModel
from train import train

from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import random


def set_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--text_dir',type=str,default= '../WebOfScience_Dataset/WOS5736/X.txt')
    parser.add_argument('--y_dir',type=str,default= '../WebOfScience_Dataset/WOS5736/Y.txt')
    parser.add_argument('--y1_dir',type=str,default= '../WebOfScience_Dataset/WOS5736/YL1.txt')
    parser.add_argument('--y2_dir',type=str,default= '../WebOfScience_Dataset/WOS5736/YL2.txt')
    parser.add_argument('--max_freq',type=int,default= 1)
    parser.add_argument('--train_batch_size',type=int,default= 8)
    parser.add_argument('--dev_batch_size',type=int,default= 8)
    parser.add_argument('--learning_rate',type=float,default=3e-4)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--num_train_epochs',type=int,default=20)
    args = parser.parse_args()


    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = process_data(args.text_dir,args.y_dir,args.y1_dir,args.y2_dir)
    data= preprocess(data)
    args.num_labels=len(data['Y1'].unique())

    train_data,dev_data,test_data = split_data(data)

    print('building vocab.....')
    id2token,token2id = build_vocab(train_data,args.max_freq)
    print('vocab builded!!')

    print('tokenizing....')
    tokenized_train_abstract = tokenize(train_data,token2id)
    tokenized_dev_abstract = tokenize(dev_data,token2id)
    tokenized_test_abstract = tokenize(test_data,token2id)
    print('tokenizing done!!!')

    train_dataset = PrmDataset(tokenized_train_abstract,train_data)
    dev_dataset = PrmDataset(tokenized_dev_abstract,dev_data)
    test_dataset = PrmDataset(tokenized_test_abstract,test_data)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.train_batch_size)
    dev_dataloader = DataLoader(dev_dataset,shuffle=True,batch_size=args.dev_batch_size)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=args.dev_batch_size)


    model = PrmModel(vocab_size=len(id2token),
                    embedding_size= 512,
                    hidden_size=300,
                    attn_hidden_size=150,
                    cls_hidden_size=3000,
                    r_size=30,
                    num_labels=args.num_labels,
                    num_layers=1,
                    dropout=0,
                    )
    model.to(device)

    train(args,model,train_dataloader,dev_dataloader,test_dataloader)


