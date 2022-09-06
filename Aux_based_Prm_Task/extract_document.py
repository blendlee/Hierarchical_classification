
from tqdm import tqdm
from tokenizer import tokenize
import torch



def extract_data_from_auxiliary_task(model,n_splits,rate,data,token2id,id2token,attention_calc):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        new_abstracts=[]
        word_attentions=[]
        for abstract,gold_label in tqdm(zip(data['abstract'],data['Y'])):

            tokenized_abstract = tokenize(abstract,token2id)
            segmented_doc = make_segments(n_splits,tokenized_abstract)

            #reset n_splits : there is probability for document length smaller than n_splits
            n_splits = len(segmented_doc)

            num_unnecessary_segments = n_splits-round(rate*n_splits)

            segment_logits={}
            attention_label=[]
            for idx,segment in enumerate(segmented_doc):

                logits,attention_scores = model(torch.tensor(segment).to(device)) #logits : (num_categories,1,n_class)
                attention_scores = torch.cat(attention_scores,dim=0)

                segment_logit = float(logits[int(gold_label)][:,1])
                segment_logits[idx]=segment_logit
                sorted_segment_logits = sorted(segment_logits.items(),key=lambda x : x[1])[num_unnecessary_segments:]
                
                if attention_calc=='sum':
                    attention_score = torch.sum(attention_scores,dim=0) # attention_scores : (n_categories, r_size,length) 
                    attention_label.append(attention_score) # attention_score : (r_size,length) 

                elif attention_calc=='gold':
                    attention_score = attention_scores[int(gold_label)]*model.num_categories
                    attention_label.append(attention_score) # attention_score : (r_size,length) 

                elif attention_calc=='pred':
                    pred_label =  int(torch.cat(logits,dim=0).squeeze(1)[:,1].argmax())
                    attention_score = attention_scores[int(pred_label)]*model.num_categories
                    attention_label.append(attention_score) # attention_score : (r_size,length) 

                

            #logit 낮은 segment idx 추출
            unnecessary_segments_idx=[]
            for i in range(num_unnecessary_segments):
                unnecessary_segments_idx.append(sorted_segment_logits[i][0])

            # 순서대로 정리된 top-k segment와 word-level attention 추가 
            new_doc=[]
            word_att=[]
            for i in range(n_splits):
                if i not in unnecessary_segments_idx:
                    new_doc += segmented_doc[i]
                    word_att += attention_label[i].tolist()

            #decode
            new_abstract=''
            for token in new_doc:
                new_abstract += id2token[str(token)]
                new_abstract += ' '
            
            new_abstracts.append(new_abstract)
            word_attentions.append(word_att)
    data['extracted_abstracts'] = new_abstracts
    data['word_level_attentions'] = word_attentions
    return data






def make_segments(n,tokenized_abstract):
    
    length = len(tokenized_abstract)
    if length <= n :
        return [[token] for token in tokenized_abstract]

    n_tokens_per_segment = length//n+1
    num_overflowing_tokens = length%n

    segment_id=-1
    segmented_doc=[[] for _ in range(n)]
    cnt=0
    for token in tokenized_abstract:
        cnt+=1
        if cnt ==n_tokens_per_segment:
            cnt=0
            segment_id+=1
            if segment_id == num_overflowing_tokens:
                n_tokens_per_segment-=1
            
        segmented_doc[segment_id].append(token)
        
    
    return segmented_doc
