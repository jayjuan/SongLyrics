# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:37:14 2022

@author: jjuan
"""

import pandas as pd

data = pd.read_csv('tcc_ceds_music.csv')


# FUNCTION TO PREPROCESS TEXT
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
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
        print(text)
    return tmp
# END FUNCTION



# CLEAN DATA
import nltk
import ast
nltk.download('words')

dict_ = {'日本': '19 日本',
         'له': 'إستعداد له',
         'لسنا': 'القادم لسنا',
         'غيتس': 'بيل غيتس',
         'على': 'على إستعداد',
         'بيل': 'بيل غيتس',
         'الوباء': 'الوباء القادم',
         'إستعداد': 'إستعداد له',
         'és': 'koronavírus és',
         'állnak': 'kik állnak',
         'zu': 'könig zu',
         'zero': 'agenda zero'}
tmp = pd.DataFrame()
words = set(nltk.corpus.words.words())
f = lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words)
cols = data.select_dtypes(object).columns
tmp[cols] = data[cols].applymap(f)

new_string = ' '.join(w for w in nltk.wordpunct_tokenize(data['lyrics'][18518]) \
             if w.lower() in words or not w.isalpha())
nan_in_col  = tmp['track_name'].isnull()
# END CLEAN


# TURN LYRICS INTO LIST - ONLY RUN ONCE
# with open('lyrics_in_list.txt', 'w', encoding='utf-8') as f:
#     for text in df['lyrics']:
#         f.write(text)
#         f.write('\n')
# END TURN

# READ LYRICS TEXT AND APPLY PREPROCESSING
import os
os.getcwd()
with open('lyrics_in_list.txt', 'r', encoding='utf-8') as fr:
        lyrics = fr.read()
lyrics = lyrics.split('\n')
# lyrics_processed_porter_stemmer = preprocess(lyrics)
lyrics_processed_remove_non_english = preprocess(lyrics)
# END READ


# CREATE CSV FOR STEMMED WORD
# for i in lyrics_processed_porter_stemmer:    
#     with open('lyrics_processed_2.csv', 'w', encoding='utf-8') as fw:
#         fw.write(i)
#         fw.write('\n')
# END CREATE


# ENCODES TARGET USING LABELENCODER
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target = le.fit_transform(data['genre'])
# END ENCODE

# CREATES NEW DATAFRAME OF COMBINED LYRICS AND TARGET
target_df = pd.DataFrame(target, columns=['Target'])
lyrics_df = pd.DataFrame(lyrics_processed_remove_non_english , columns=['Lyrics'])
lyrics_df['Release_Date'] = data.loc[data['release_date'] > 1000]['release_date']
final_df = lyrics_df.join(target_df)
final_df = final_df.join(data['genre'])
final_df = final_df.dropna() # REMOVE DATES THAT ARE NAN
# END CREATE


# REMOVE EMPTY DATA
import numpy as np
final_df['Lyrics'].replace('', np.nan, inplace=True)
final_df.dropna(subset=['Lyrics'], inplace=True)
# END REMOVE


# MODEL
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection._univariate_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(final_df['Lyrics'])
Y = final_df['Target']
# END MODEL


# COUNT VECTORIZER
countvectorizer = CountVectorizer()
count = countvectorizer.fit_transform(lyrics_processed_remove_non_english)

count_tokens = countvectorizer.get_feature_names_out()
# END COUNT


# SPLIT INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y,
                                                    test_size=0.4, 
                                                    random_state=42)
# END SPLIT


# EVALUATE MODEL
ch2 = SelectKBest(chi2, k=1000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

clf = MultinomialNB()
# clf = LinearSVC()
#clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy: %0.3f" % metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, target_names=pd.unique(data['genre'])))
# END EVALUATE

final_df.to_csv('20223011 Processed DataFrame.csv', index=False)
