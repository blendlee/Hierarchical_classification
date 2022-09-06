import pandas as pd
import re
from sklearn.model_selection import ShuffleSplit

def process_data(text_dir,y_dir,y1_dir,y2_dir):

    text_file = open(text_dir,'r')
    y_file = open(y_dir,'r')
    y1_file = open(y1_dir,'r')
    y2_file = open(y2_dir,'r')

    text=text_file.readlines()
    y=y_file.readlines()
    y1=y1_file.readlines()
    y2=y2_file.readlines()

    text_file.close()
    y_file.close()
    y1_file.close()
    y2_file.close()

    for i in range(len(text)-1):
        text[i] = text[i].strip()
        y[i]=y[i].strip()
        y1[i]=y1[i].strip()
        y2[i]=y2[i].strip()
    
    df = pd.DataFrame({'abstract':text,'Y1':y1,'Y2':y2,'Y':y})

    return df

def preprocess(df):
    corpus=[]
    for text in list(df['abstract']):
        text = re.sub('[-=+▲#/\:^@*\"※~ㆍ!』|\(\)\[\]`\…》\·\n]','',text)
        text = re.sub('[,.]','',text)
        corpus.append(text)

    df['abstract'] = corpus

    return df

def split_data(data):
    train_sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    test_sss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

    indices = range(len(data))
    for train_index, test_val_index in train_sss.split(indices):
        train_index, test_val_index=train_index, test_val_index
    train_data=data.iloc[train_index].reset_index(drop=True)
    test_val_data=data.iloc[test_val_index].reset_index(drop=True)

    test_val_indices =range(len(test_val_data))
    for val_index, test_index in test_sss.split(test_val_indices):
        val_index, test_index=val_index, test_index

    val_data = test_val_data.iloc[val_index].reset_index(drop=True)
    test_data = test_val_data.iloc[test_index].reset_index(drop=True)

    return train_data,val_data,test_data