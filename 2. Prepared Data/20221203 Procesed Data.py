# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:41:36 2022

@author: jjuan
"""

import pandas as pd

df = pd.read_csv('tcc_ceds_music.csv', usecols=['lyrics'])

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

words = set(nltk.corpus.words.words())
stop_words=set(stopwords.words("english"))
porter = PorterStemmer()
def preprocess(texts):
    tmp = []
    for n, j in enumerate(texts):
        text = j
        text = word_tokenize(text)
        text = [word.lower() for word in text]
        text = [word for word in text if word not in stop_words]
        text = [word for word in text if word.isalpha()]
        text = [porter.stem(word) for word in text]
        text = " ".join(text)
        text = " ".join(w for w in nltk.wordpunct_tokenize(text) \
                     if w.lower() in words or not w.isalpha())
        tmp.append(text)
        print(f'{n*100/len(texts):.2f}%')
    print('****\nDone!\n****')
    return tmp

processed_lyrics= preprocess(df['lyrics']) # returns a list

df_processed = pd.DataFrame()
df_processed['Lyrics'] = processed_lyrics
df_processed = df_processed[df_processed['Lyrics']!='']


print(len(df['lyrics'][0]))
print(len(df_processed['Lyrics'][0]))

agg_ori = 0
for i in df['lyrics']:
    agg_ori += len(i)
print(agg_ori/len(df['lyrics']))

for i in df_processed['Lyrics']:
    agg_ori += len(i)
print(agg_ori/len(df_processed['Lyrics']))