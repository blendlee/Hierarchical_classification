import pandas as pd
import json
import os
import re


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

def split_data(data,partition_pth):
    pass