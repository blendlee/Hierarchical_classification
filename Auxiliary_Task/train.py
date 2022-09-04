
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import trange,tqdm

import os
import os.path
import torch
import torch.nn as nn

def get_metric_result(preds,labels):

    results=[]

    for i in range(5):
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
        total_logits=[[],[],[],[],[]]
        total_aspects=[]
        train_loss=0
        aspect_train_loss=[0,0,0,0,0]

        for _, batch in enumerate(batch_iterator):

            optimizer.zero_grad()

            input = batch[0].to(device)
            aspects= batch[1].T.to(device)

            logits,attention_score = model(input)

            #print(logits)
            #aspect loss 다시보기
            aspect_losses=[criterion(logits[i][:,1],aspects[i].float()) for i in range(5)]
            aspect_train_loss = [aspect_train_loss[i]+aspect_losses[i] for i in range(5)]
            
            #print(aspect_losses)
            total_loss = sum(aspect_losses)
            #print(total_loss)
            train_loss += total_loss

            #print('backward start')
            total_loss.backward()
            optimizer.step()

            #print('backward done')
            for i in range(5):
                total_logits[i].append(logits[i])
            total_aspects.append(aspects)
            #print('collecting logits done!')

        for i in range(5):
            total_logits[i] = torch.cat(total_logits[i],dim=0)


        train_loss /= total_steps
        aspect_train_loss = [loss/total_steps for loss in aspect_train_loss]
        total_aspects = torch.cat(total_aspects,dim=1)
        train_results = get_metric_result(total_logits,total_aspects)

        print(f'-------epoch {epoch+1} train result-------')
        print(f'Training loss : {train_loss}')
        print(f'Training aspects loss : {aspect_train_loss}')
        print(f'Nudity      >> accuracy : {train_results[0][0]}, precision : {train_results[0][1]} , f1-score : {train_results[0][2]} , recall : {train_results[0][3]}')
        print(f'Violence    >> accuracy : {train_results[1][0]}, precision : {train_results[1][1]} , f1-score : {train_results[1][2]} , recall : {train_results[1][3]}')
        print(f'Profanity   >> accuracy : {train_results[2][0]}, precision : {train_results[2][1]} , f1-score : {train_results[2][2]} , recall : {train_results[2][3]}')
        print(f'Substance   >> accuracy : {train_results[3][0]}, precision : {train_results[3][1]} , f1-score : {train_results[3][2]} , recall : {train_results[3][3]}')
        print(f'frightening >> accuracy : {train_results[4][0]}, precision : {train_results[4][1]} , f1-score : {train_results[4][2]} , recall : {train_results[4][3]}')
        print('\n\n')


        evaluate(model,dev_dataloader,epoch,'Validation')
        evaluate(model,test_dataloader,epoch,'Test')


        if not os.path.isdir('models'):
            os.mkdir('models')
        torch.save(model.state_dict(),f'models/{epoch+1} epoch model.pth')





def evaluate(model,eval_dataloader,epoch,status):
    
    eval_iterator = tqdm(eval_dataloader, desc="Iteration")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.BCELoss()


    total_steps = len(eval_iterator)
    total_logits=[[],[],[],[],[]]
    total_aspects=[]
    eval_loss=0
    aspect_eval_loss=[0,0,0,0,0]


    with torch.no_grad():
        model.eval()
        for _,batch in enumerate(eval_iterator):

            input = batch[0].to(device)
            aspects = batch[1].T.to(device)

            logits,attention_score = model(input)

            aspect_losses=[criterion(logits[i][:,1],aspects[i].float()) for i in range(5)]
            aspect_eval_loss = [aspect_eval_loss[i]+aspect_losses[i] for i in range(5)]
            
            total_loss = sum(aspect_losses)
            eval_loss += total_loss


            for i in range(5):
                total_logits[i].append(logits[i])
            total_aspects.append(aspects)

        for i in range(5):
            total_logits[i] = torch.cat(total_logits[i],dim=0)

        eval_loss /= total_steps
        aspect_eval_loss = [loss/total_steps for loss in aspect_eval_loss]
        total_aspects = torch.cat(total_aspects,dim=1)
        eval_results = get_metric_result(total_logits,total_aspects)

        print(f'-------epoch {epoch+1} {status} result-------')
        print(f'Evaluation loss : {eval_loss}')
        print(f'Evaluation aspects loss : {aspect_eval_loss}')
        print(f'Nudity      >> accuracy : {eval_results[0][0]}, precision : {eval_results[0][1]} , f1-score : {eval_results[0][2]} , recall : {eval_results[0][3]}')
        print(f'Violence    >> accuracy : {eval_results[1][0]}, precision : {eval_results[1][1]} , f1-score : {eval_results[1][2]} , recall : {eval_results[1][3]}')
        print(f'Profanity   >> accuracy : {eval_results[2][0]}, precision : {eval_results[2][1]} , f1-score : {eval_results[2][2]} , recall : {eval_results[2][3]}')
        print(f'Substance   >> accuracy : {eval_results[3][0]}, precision : {eval_results[3][1]} , f1-score : {eval_results[3][2]} , recall : {eval_results[3][3]}')
        print(f'frightening >> accuracy : {eval_results[4][0]}, precision : {eval_results[4][1]} , f1-score : {eval_results[4][2]} , recall : {eval_results[4][3]}')
        print('\n\n')





