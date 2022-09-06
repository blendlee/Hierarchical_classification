import argparse
from preprocess import  preprocess, process_data,split_data
from tokenizer import build_vocab,tokenize
from dataset import AuxBasedPrmDataset
from model import AuxBasedPrmModel,AuxModel
from extract_document import extract_data_from_auxiliary_task
from train import train

from torch.utils.data import DataLoader
import torch





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
    parser.add_argument('--aux_model_dir',type=str,default='../Auxiliary_Task/models/9_epoch_Aux_model.pth')
    parser.add_argument('--n_splits',type=int,default=30)
    parser.add_argument('--segment_rate',type=float,default=0.9)
    parser.add_argument('--lambda_rate',type=float,default=0.1)
    parser.add_argument('--r_size',type=int,default=30)
    args = parser.parse_args()


    #set_seed()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = process_data(args.text_dir,args.y_dir,args.y1_dir,args.y2_dir)
    data= preprocess(data)
    args.num_labels=len(data['Y1'].unique())

    train_data,dev_data,test_data = split_data(data)
    print('building vocab.....')
    id2token,token2id = build_vocab(train_data,args.max_freq)
    print('vocab builded!!')


    #For Auxiliary Task
    auxiliary_model = AuxModel(vocab_size=len(id2token),
                                embedding_size= 512,
                                hidden_size=300,
                                attn_hidden_size=150,
                                cls_hidden_size=3000,
                                r_size=args.r_size,
                                num_layers=1,
                                dropout=0,
                                num_categories=args.num_categories
                                )
    auxiliary_model.load_state_dict(args.aux_model_dir)
    auxiliary_model.to(device)

    #Auxiliary Task
    print('Extracting document & Data by Auxiliary Model.......')
    train_data = extract_data_from_auxiliary_task(auxiliary_model,args.n_splits,args.segment_rate,train_data,token2id,id2token,'sum')
    dev_data = extract_data_from_auxiliary_task(auxiliary_model,args.n_splits,args.segment_rate,dev_data,token2id,id2token,'sum')
    test_data = extract_data_from_auxiliary_task(auxiliary_model,args.n_splits,args.segment_rate,test_data,token2id,id2token,'sum')
    print('Extracting done!!!!!')

    #Primary Task
    print('tokenizing....')
    tokenized_train_abstract = tokenize(list(train_data['extracted_abstracts']),token2id)
    tokenized_dev_abstract = tokenize(list(dev_data['extracted_abstracts']),token2id)
    tokenized_test_abstract = tokenize(list(test_data['extracted_abstracts']),token2id)
    print('tokenizing done!!!')


    train_dataset = AuxBasedPrmDataset(tokenized_train_abstract,train_data)
    dev_dataset = AuxBasedPrmDataset(tokenized_dev_abstract,dev_data)
    test_dataset = AuxBasedPrmDataset(tokenized_test_abstract,test_data)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.train_batch_size)
    dev_dataloader = DataLoader(dev_dataset,shuffle=True,batch_size=args.dev_batch_size)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=args.dev_batch_size)



    primary_model = AuxBasedPrmModel(vocab_size=len(id2token),
                                    embedding_size= 512,
                                    hidden_size=300,
                                    attn_hidden_size=150,
                                    cls_hidden_size=3000,
                                    r_size=args.r_size,
                                    num_labels=args.num_labels,
                                    num_layers=1,
                                    dropout=0,
                                    )
    primary_model.to(device)


    train(args,primary_model,train_dataloader,dev_dataloader,test_dataloader)


