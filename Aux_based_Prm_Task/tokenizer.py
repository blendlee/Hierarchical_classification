from torchtext.data.utils import get_tokenizer
from collections import Counter

import json
import os
import os.path


def build_vocab(train_data,frequency):
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()
    corpus=list(train_data['abstract'])

    for script in corpus:
        tokens = tokenizer(script.lower())
        counter.update(tokens)

    id=0
    special_tokens = ['<pad>','<unk>']
    id2token = {}
    token2id = {}
    for token in special_tokens:
        id2token[id] = token
        token2id[token] = id
        id+=1

    for token,count in counter.most_common():
        if count >= frequency:
            id2token[id] = token
            token2id[token] = id
            id+=1

    if not os.path.isdir('vocab'):
        os.makedirs('vocab')

    with open('vocab/id2token.json', 'w') as f:
        json.dump(id2token, f)
    with open('vocab/token2id.json', 'w') as f:
        json.dump(token2id, f)

    return id2token,token2id


def tokenize(abstracts, token2id):

    tokenizer = get_tokenizer("basic_english")

    if isinstance(abstracts,str) :
        tokens = tokenizer(abstracts.lower())
        tokenized_sentence=[]
        for token in tokens:
            if token not in token2id:
                token='<unk>'
            tokenized_sentence.append(token2id[token])

        return tokenized_sentence


    else:
        tokenized_abstracts=[]
        max_len = 0
        for abstract in abstracts:
            tokens = tokenizer(abstract.lower())
            tokenized_sentence=[]

            for token in tokens:
                if token not in token2id:
                    token='<unk>'
                tokenized_sentence.append(token2id[token])

            max_len = max(max_len,len(tokenized_sentence))
            tokenized_abstracts.append(tokenized_sentence)
        return pad_sequence(max_len,tokenized_abstracts),max_len


def pad_sequence(max_len,tokenized_abstracts):

    for idx,tokenized_sentence in enumerate(tokenized_abstracts):
        if len(tokenized_sentence) < max_len:
            tokenized_abstracts[idx] += [0]*(max_len-len(tokenized_sentence))
                    
    return tokenized_abstracts

def pad_sequence_for_attention(max_len,attentions):

    size= len(attentions[0][0])
    for idx,att in enumerate(attentions):
        if len(att) < max_len:
            attentions[idx] += [[0]*size for _ in range(max_len-len(att))]
    return attentions