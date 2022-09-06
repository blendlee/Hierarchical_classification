from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import trange,tqdm

import os
import os.path
import torch
import torch.nn as nn

def get_metric_result(preds,labels):

    labels = labels.detach().cpu()
    preds = preds.detach().cpu()
    accuracy = (labels == preds).sum() / len(labels)
    precision = precision_score(labels,preds,average='macro')
    f1 = f1_score(labels,preds,average='macro')
    recall = recall_score(labels,preds,average='macro')
    
    results=[accuracy,precision,f1,recall]

    return results

def train(args,model,train_dataloader,dev_dataloader,test_dataloader):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:

        batch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        criterion = nn.CrossEntropyLoss()

        total_steps = len(batch_iterator)
        all_logits=[]
        all_labels=[]
        train_loss=0


        for _, batch in enumerate(batch_iterator):

            optimizer.zero_grad()

            input = batch[0].to(device)
            labels= batch[1].type(torch.LongTensor).to(device)

            logits,attention_score = model(input)

            loss = criterion(logits,labels)
            train_loss += loss

            loss.backward()
            optimizer.step()
            
            all_logits.append(logits)
            all_labels.append(labels)

        all_logits = torch.cat(all_logits,dim=0)
        all_labels = torch.cat(all_labels,dim=0)
        preds = torch.argmax(all_logits,dim=1)

        train_loss /= total_steps

        train_results = get_metric_result(preds,all_labels)

        print(f'-------epoch {epoch+1} train result-------')
        print(f'Training loss : {train_loss}')
        print(f'Domain   >> accuracy : {train_results[0]}, precision : {train_results[1]} , f1-score : {train_results[2]} , recall : {train_results[3]}')
        print('\n\n')


        evaluate(model,dev_dataloader,epoch,'Validation')
        evaluate(model,test_dataloader,epoch,'Test')


        if not os.path.isdir('models'):
            os.mkdir('models')
        torch.save(model.state_dict(),f'models/{epoch+1}_epoch_primary_model.pth')





def evaluate(model,eval_dataloader,epoch,status):
    
    eval_iterator = tqdm(eval_dataloader, desc="Iteration")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()


    total_steps = len(eval_iterator)
    all_logits=[]
    all_labels=[]
    eval_loss=0


    with torch.no_grad():
        model.eval()
        for _,batch in enumerate(eval_iterator):

            input = batch[0].to(device)
            labels= batch[1].type(torch.LongTensor).to(device)

            logits,attention_score = model(input)


            loss = criterion(logits,labels)
            eval_loss += loss


            all_logits.append(logits)
            all_labels.append(labels)

        eval_loss /= total_steps
        all_logits = torch.cat(all_logits,dim=0)
        all_labels = torch.cat(all_labels,dim=0)
        preds = torch.argmax(all_logits,dim=1)

        eval_results = get_metric_result(preds,all_labels)

        print(f'-------epoch {epoch+1} {status} result-------')
        print(f'Evaluation loss : {eval_loss}')
        print(f'Domain   >> accuracy : {eval_results[0]}, precision : {eval_results[1]} , f1-score : {eval_results[2]} , recall : {eval_results[3]}')
        print('\n\n')




