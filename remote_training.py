# This .py file intend the remote computation of time consuming training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
import time
import datetime as datetime

## My script : 
from function import *

# Specific libraries : 
import nltk
from bs4 import BeautifulSoup


from sklearn.feature_extraction.text import CountVectorizer # BoW
from sklearn.feature_extraction.text import TfidfVectorizer # Tfidf


from sklearn.metrics import f1_score,precision_score,recall_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn import metrics

def reforme_tags_processed_text(data):
    tags = []
    processed_text = []
    for indx in range(data.shape[0]):
        new_tags = []
        new_processed = []
        
        split = data['Tags'][indx].split(',')
        for nb_tags in range(data['nb_tags'][indx]):    
            to_append = re.sub('[\[\]\'\"!*+-]','',split[nb_tags]).replace('\\','').replace(' ','')
            if not to_append in ['',' '] : new_tags.append(to_append)
        tags.append(new_tags)
        
        text = data['processed_text'][indx].split(',')
        for i in range(len(text)):
            to_append = re.sub('[\[\]\'\"!*+-:.]','',text[i]).replace('\\','').replace(' ','')
            if not to_append in ['',' '] : new_processed.append(to_append)
        processed_text.append(new_processed)
        
    data['Tags'] = tags
    data['processed_text'] = processed_text
    return data


def get_all_tags(tags):
    res = []
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            res.append(tags[i][j])
            
    return pd.Series(res)


root_dir = '/Users/jeremynadal/Documents/Formation OC IML/P5-API/'
root_dir = '/net/cremi/jnadal/categorize_question_API'
input_dir = root_dir + 'inputs/'
png_dir = root_dir + 'pngs/'

data = pd.read_csv(input_dir+'processed_dataset.csv')



data = reforme_tags_processed_text(data)
tags = get_all_tags(data['Tags'])

unique_tags = np.unique(tags)

for tag in unique_tags : 
    data[tag] = 0
    
for idx in data.index:
    for id_tag in range(data['nb_tags'][idx]):
        data[data['Tags'][idx][id_tag]][idx] = 1
        

bow_vectorizer = CountVectorizer(tokenizer = lambda x: x,
                                 preprocessor = lambda x: x,
                                 lowercase = False,
                                 max_features = 1000,
                                 binary = True,
                                 max_df = 0.9
                                 )  
bow_X = bow_vectorizer.fit_transform(data['processed_text'])

tfidf_vectorizer = TfidfVectorizer(tokenizer = lambda x: x,
                                   preprocessor = lambda x: x,
                                   lowercase = False,
                                   max_features = 1000,
                                   max_df = 0.9
                                   )  
tfidf_X = tfidf_vectorizer.fit_transform(data['processed_text'])

X_train_bow, X_test_bow, y_train_multi, y_test_multi = train_test_split(bow_X.toarray(), 
                                                                     data[unique_tags], 
                                                                     test_size=0.2, 
                                                                     random_state=42)
X_train_tfidf, X_test_tfidf, y_train_tags, y_test_tags = train_test_split(tfidf_X.toarray(), 
                                                                            data['Tags'], 
                                                                            test_size=0.2, 
                                                                            random_state=42)

def cosine_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    assert x.shape == y.shape , 'x and y doesnt have same shape'
    assert len(x.shape)==2 , 'x and y must be matrixes [nb_samp,x], if only 1 sample use : np.reshape(1,-1)'
    cosin = []
    for idx in range(x.shape[0]):
        if (np.dot(x[idx], x[idx]) !=0 and np.dot(y[idx], y[idx]) != 0 ) :
            cosin.append(np.dot(x[idx], y[idx]) / (np.sqrt(np.dot(x[idx], x[idx])) * np.sqrt(np.dot(y[idx], y[idx]))))
        elif (np.dot(x[idx], x[idx]) ==0 and np.dot(y[idx], y[idx]) == 0 ) : 
            cosin.append(1)
        else:
            cosin.append(-1)
    return np.mean(cosin)

def print_metrics(y_true, pred):
    '''Prints and return a summary of results'''
    y_true = np.array(y_true)
    pred = np.array(pred)
    assert y_true.shape == pred.shape, 'arrays doesnt have same shape'
    
    res = [metrics.accuracy_score(y_true, pred),
           metrics.hamming_loss(y_true,pred),
           precision_score(y_true, pred, average='micro'),
           recall_score(y_true, pred, average='micro'),
           f1_score(y_true, pred, average='micro'),
           cosine_similarity(y_true,pred)]
    
    print("Accuracy :{:.3}\nHamming loss :{:.4}\n\nMicro-averaged quality metrics :\nPrecision :{:.3}\nRecall :{:.3}\nF1-score :{:.3}\nCosine similarity : {:.3}".format(*res))
    return res


start = time.process_time()
#classifier = OneVsRestClassifier(xgboost.XGBClassifier(random_state=42))
classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.001, penalty='l1'))

classifier.fit(X_train_bow, y_train_multi)
predictions = classifier.predict(X_test_bow)


res = print_metrics(y_test_multi, predictions)
tps = time.process_time() - start

classifier_comparison = pd.DataFrame(columns = ['model','vectorizer','acc','hamm','precision','recall','F1','cosine','tps'])


rf = {'model':'xgboost',
      'vectorizer':'bow',
      'acc':res[0],
      'hamm':res[1],
      'precision':res[2],
      'recall':res[3],
      'F1':res[4],
      'cosine':res[5],
      'tps':tps}
classifier_comparison = classifier_comparison.append(rf,ignore_index=True)
classifier_comparison.to_csv(input_dir+'xgboost.csv')

start = time.process_time()
#classifier = OneVsRestClassifier(xgboost.XGBClassifier(random_state=42))
classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.001, penalty='l1'))

classifier.fit(X_train_tfidf, y_train_multi)
predictions = classifier.predict(X_test_bow)


res = print_metrics(y_test_multi, predictions)
tps = time.process_time() - start


rf = {'model':'xgboost',
      'vectorizer':'tfidf',
      'acc':res[0],
      'hamm':res[1],
      'precision':res[2],
      'recall':res[3],
      'F1':res[4],
      'cosine':res[5],
      'tps':tps}
classifier_comparison = classifier_comparison.append(rf,ignore_index=True)
classifier_comparison.to_csv(input_dir+'xgboost.csv')


