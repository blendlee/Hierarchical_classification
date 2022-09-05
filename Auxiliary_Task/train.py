
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import trange,tqdm

import os
import os.path
import torch
import torch.nn as nn

def get_metric_result(preds,labels):

    results=[]

    for i in range(len(labels)):
        _,max_indices = torch.max(preds[i],1)
        label = labels[i].detach().cpu()
        pred = max_indices.detach().cpu()

        accuracy = (label == pred).sum() / len(label)
        precision = precision_score(label,pred,average='macro')
        f1 = f1_score(label,pred,average='macro')
        recall = recall_score(label,pred,average='macro')
        

        results.append([accuracy,precision,f1,recall])

    return results

def train(args,model,train_dataloader,dev_dataloader,test_dataloader):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:

        batch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        criterion = nn.BCELoss()

        total_steps = len(batch_iterator)
        all_logits=[[] for _ in range(args.num_categories)]
        all_categories=[]
        train_loss=0
        category_train_loss=[0]*args.num_categories

        for _, batch in enumerate(batch_iterator):

            optimizer.zero_grad()

            input = batch[0].to(device)
            categories= batch[1].T.to(device)

            logits,attention_score = model(input)

            category_losses = [criterion(logits[i][:,1],categories[i].float()) for i in range(args.num_categories)]
            category_train_loss = [category_train_loss[i]+category_losses[i] for i in range(args.num_categories)]
            

            total_loss = sum(category_losses)
            train_loss += total_loss

            #print('backward start')
            total_loss.backward()
            optimizer.step()

            #print('backward done')
            for i in range(args.num_categories):
                all_logits[i].append(logits[i])
            all_categories.append(categories)

        for i in range(args.num_categories):
            all_logits[i] = torch.cat(all_logits[i],dim=0)


        train_loss /= total_steps
        category_train_loss = [loss/total_steps for loss in category_train_loss]
        all_categories = torch.cat(all_categories,dim=1)
        train_results = get_metric_result(all_logits,all_categories)

        print(f'-------epoch {epoch+1} train result-------')
        print(f'Training loss : {train_loss}')
        print(f'Training categories loss : {category_train_loss}')
        for i in range(args.num_categories):
            print(f'Category {i+1}  >> accuracy : {train_results[i][0]}, precision : {train_results[i][1]} , f1-score : {train_results[i][2]} , recall : {train_results[i][3]}')
        print('\n\n')


        evaluate(args,model,dev_dataloader,epoch,'Validation')
        evaluate(args,model,test_dataloader,epoch,'Test')


        if not os.path.isdir('models'):
            os.mkdir('models')
        torch.save(model.state_dict(),f'models/{epoch+1}_epoch_Aux_model.pth')





def evaluate(args,model,eval_dataloader,epoch,status):
    
    eval_iterator = tqdm(eval_dataloader, desc="Iteration")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.BCELoss()


    total_steps = len(eval_iterator)
    all_logits=[[] for _ in range(args.num_categories)]
    all_categories=[]
    eval_loss=0
    category_train_loss=[0]*args.num_categories


    with torch.no_grad():
        model.eval()
        for _,batch in enumerate(eval_iterator):

            input = batch[0].to(device)
            categories = batch[1].T.to(device)

            logits,attention_score = model(input)

            category_losses=[criterion(logits[i][:,1],categories[i].float()) for i in range(args.num_categories)]
            category_eval_loss = [category_eval_loss[i]+category_losses[i] for i in range(args.num_categories)]
            
            total_loss = sum(category_losses)
            eval_loss += total_loss


            for i in range(args.num_categories):
                all_logits[i].append(logits[i])
            all_categories.append(categories)

        for i in range(args.num_categories):
            all_logits[i] = torch.cat(all_logits[i],dim=0)

        eval_loss /= total_steps
        category_eval_loss = [loss/total_steps for loss in category_eval_loss]
        all_categories = torch.cat(all_categories,dim=1)
        eval_results = get_metric_result(all_logits,all_categories)

        print(f'-------epoch {epoch+1} {status} result-------')
        print(f'Evaluation loss : {eval_loss}')
        print(f'Evaluation categories loss : {category_eval_loss}')
        for i in range(args.num_categories):
            print(f'Category {i+1}  >> accuracy : {eval_results[i][0]}, precision : {eval_results[i][1]} , f1-score : {eval_results[i][2]} , recall : {eval_results[i][3]}')
        print('\n\n')





