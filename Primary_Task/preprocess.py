import pandas as pd
import json
import os
import re


def process_data(meta_data_pth,mpaa_data_pth,script_pth):

    metadata=pd.read_csv(meta_data_pth)
    mpaa_components=pd.read_csv(mpaa_data_pth)
    files = os.listdir(script_pth)

    def slice(x):
        return x[2:-2]
    mpaa_components['imdbID']=mpaa_components['imdbID'].apply(slice)
    mpaa_components = mpaa_components.dropna(axis=0)

    comp_id = list(mpaa_components['imdbID'])
    meta_id = list(metadata['IMDB_ID'])
    file_id = [file[:-4] for file in files]
    data_id = list(set(comp_id)&set(meta_id)&set(file_id))

    mpaa_components=mpaa_components.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)
    comp_idx=[]
    meta_idx=[]

    for i in range(len(mpaa_components)):
        if mpaa_components['imdbID'][i] in data_id:
            comp_idx.append(i)
    for j in range(len(metadata)):
        if metadata['IMDB_ID'][j] in data_id:
            meta_idx.append(j)


    mpaa_components  = mpaa_components.iloc[comp_idx].sort_values('imdbID').reset_index(drop=True)
    meta_data = metadata.iloc[meta_idx].sort_values('IMDB_ID').reset_index(drop=True)

    df1 = meta_data.loc[:,['IMDB_ID','Title','Genre','MPAA_rating','Year']]
    df2 = mpaa_components.loc[:,['imdbID','v_level','n_level','p_level','a_level','f_level']]
    dataset = pd.concat([df1,df2],axis=1)

    dataset.rename(columns = {'n_level' : 'sex'}, inplace = True)
    dataset.rename(columns = {'v_level' : 'violence'}, inplace = True)
    dataset.rename(columns = {'p_level' : 'profanity'}, inplace = True)
    dataset.rename(columns = {'a_level' : 'substance'}, inplace = True)
    dataset.rename(columns = {'f_level' : 'frightening'}, inplace = True)
    dataset = dataset.drop('imdbID',axis=1)

    dataset.loc[dataset['MPAA_rating']=='G',['rating']]=0
    dataset.loc[dataset['MPAA_rating']=='PG',['rating']]=0
    dataset.loc[dataset['MPAA_rating']=='PG-13',['rating']]=1
    dataset.loc[dataset['MPAA_rating']=='R',['rating']]=2
    dataset.loc[dataset['MPAA_rating']=='NC-17',['rating']]=2

    levels=['sex','violence','profanity','substance','frightening']
    for level in levels:
        dataset.loc[dataset[level]=='None',[level+'_level']]=0
        dataset.loc[dataset[level]=='Mild',[level+'_level']]=0
        dataset.loc[dataset[level]=='Moderate',[level+'_level']]=1
        dataset.loc[dataset[level]=='Severe',[level+'_level']]=1


    return dataset

def load_script_with_preprocessed(dataset,script_pth):

    corpus=[]
    for i in range(len(dataset)):
        text = open(script_pth+'/'+dataset['IMDB_ID'][i]+'.txt','r').readlines()
        script=''
        for sentence in text:
            script += sentence
        script = re.sub('[-=+▲#/\:^@*\"※~ㆍ!』|\(\)\[\]`\…》\·\n]','',script)
        script = re.sub('[,.]','',script)
        corpus.append(script)

    dataset['script'] = corpus

    return dataset, corpus



def split_data(data,partition_pth):

    with open(partition_pth, 'r') as f :
        json_data = json.load(f)

    train_id = json_data['train']
    dev_id = json_data['dev']
    test_id = json_data['test']

    train_idx=[]
    dev_idx=[]
    test_idx=[]

    for i in range(len(data)):
        if data['IMDB_ID'][i] in train_id:
            train_idx.append(i)
        elif data['IMDB_ID'][i] in dev_id:
            dev_idx.append(i)
        elif data['IMDB_ID'][i] in test_id:
            test_idx.append(i)

    train_data = data.iloc[train_idx].reset_index(drop=True)
    dev_data = data.iloc[dev_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)

    return train_data,dev_data,test_data