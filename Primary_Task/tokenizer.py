from torchtext.data.utils import get_tokenizer
from collections import Counter

import json
import os
import os.path


def build_vocab(corpus,frequency):
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()
    for script in corpus:
        tokens = tokenizer(script.lower())
        counter.update(tokens)

    id=0
    special_tokens = ['<pad>','<unk>','<sos>','<eos>']
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


def tokenize(data, token2id):

    tokenizer = get_tokenizer("basic_english")
    scripts = list(data['script'])
    tokenized_scripts=[]

    max_len = 0
    for script in scripts:
        tokens = tokenizer(script.lower())
        tokenized_sentence=[]

        for token in tokens:
            if token not in token2id:
                token='<unk>'
            tokenized_sentence.append(token2id[token])

        max_len = max(max_len,len(tokenized_sentence))
        tokenized_scripts.append(tokenized_sentence)

    #padding

    for idx,tokenized_sentence in enumerate(tokenized_scripts):
        if len(tokenized_sentence) < max_len:
            tokenized_scripts[idx] += [token2id['<pad>']]*(max_len-len(tokenized_sentence))
            

    return tokenized_scripts

