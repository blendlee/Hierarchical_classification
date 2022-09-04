import argparse
from preprocess import  load_script_with_preprocessed, process_data,split_data
from tokenizer import build_vocab,tokenize
from dataset import PrmDataset
from model import PrmModel
from train import train

from torch.utils.data import DataLoader
import torch





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--meta_data_pth',type=str,default= '../Deliver_dataset/meta_information/metadata.csv')
    parser.add_argument('--mpaa_data_pth',type=str,default= '../Deliver_dataset/meta_information/MPAA_components.csv')
    parser.add_argument('--script_pth',type=str,default= '../Deliver_dataset/Script')
    parser.add_argument('--partition_pth',type=str,default= '../Deliver_dataset/meta_information/partitions.json')
    parser.add_argument('--max_freq',type=int,default= 1)
    parser.add_argument('--train_batch_size',type=int,default= 8)
    parser.add_argument('--dev_batch_size',type=int,default= 8)
    parser.add_argument('--learning_rate',type=float,default=3e-4)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--num_train_epochs',type=int,default=20)
    args = parser.parse_args()


    #set_seed()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = process_data(args.meta_data_pth,args.mpaa_data_pth,args.script_pth)

    train_data,dev_data,test_data = split_data(data,args.partition_pth)

    train_data,corpus = load_script_with_preprocessed(train_data,args.script_pth)
    dev_data,_ = load_script_with_preprocessed(dev_data,args.script_pth)
    test_data,_ = load_script_with_preprocessed(test_data,args.script_pth)

    print('building vocab.....')
    id2token,token2id = build_vocab(corpus,args.max_freq)
    print('vocab builded!!')

    print('tokenizing....')
    tokenized_train_script = tokenize(train_data,token2id)
    tokenized_dev_script = tokenize(dev_data,token2id)
    tokenized_test_script = tokenize(test_data,token2id)
    print('tokenizing done!!!')

    train_dataset = PrmDataset(tokenized_train_script,train_data)
    dev_dataset = PrmDataset(tokenized_dev_script,dev_data)
    test_dataset = PrmDataset(tokenized_test_script,test_data)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.train_batch_size)
    dev_dataloader = DataLoader(dev_dataset,shuffle=True,batch_size=args.dev_batch_size)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=args.dev_batch_size)

    model = PrmModel(vocab_size=len(id2token),
                    embedding_size= 512,
                    hidden_size=300,
                    attn_hidden_size=150,
                    cls_hidden_size=3000,
                    r_size=30,
                    num_layers=1,
                    dropout=0,
                    )
    model.to(device)
    train(args,model,train_dataloader,dev_dataloader,test_dataloader)


