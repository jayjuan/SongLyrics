# Report part:
# a. Introduction (1 person)
# b. Data collection
# - Describe the text dataset (music dataset) (give a detailed account of the source, content, properties, and statistics of the dataset) (1 person)
# c. document representation methods (refer to coding part)
# d. algorithms used for the 3 system functions (refer to coding part)
# e system testing and performance (plus metrics used) (refer to coding part)
# f. Limitations and difficulties encountered (1 person)
# g. Technical decisions made and their effects on performance (1 person)
# h. Conclusion (1 person)

# Coding part: 
# 1. Be able to accurately distinguish X documents from all other types (classification)
# - document representation methods => Vector space model (bag of words, stemming, TF-IDF, tokenization, vectorization…) (3 person)
# - assessed and analysed according to accuracy, precision and recall, F1 score, confusion matrix 
# 2. Be able to automatically divide X documents into similar groups (Clustering) (2 person)
# - Assessed and analysed according to clustering tendency, number of clusters, clustering quality (2)
# 3. Be able to automatically find topics within the set of X documents (Topic modelling)
# - Assessed and analysed according to coherence and human judgement (2 person)


# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:19:28 2022

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
    for n in range(0, len(texts)):
        text = texts[n]
        text = word_tokenize(text)
        text = [word.lower() for word in text]
        text = [word for word in text if word not in stop_words]
        text = [word for word in text if word.isalpha()]
        text = [porter.stem(word) for word in text]
        text = " ".join(text)
        texts[n] = text
    return texts
# END FUNCTION

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
lyrics_processed = lyrics
lyrics_processed_porter_stemmer = preprocess(lyrics)
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
lyrics_df = pd.DataFrame(lyrics, columns=['Lyrics'])
final_df = target_df.join(lyrics_df)
# END CREATE

# SPLIT INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_df['Lyrics'], 
                                                    final_df['Target'], 
                                                    test_size=0.4, 
                                                    random_state=42)
# END SPLIT


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
count = countvectorizer.fit_transform(final_df['Lyrics'])

count_tokens = countvectorizer.get_feature_names_out()
# END COUNT


# SPLIT INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y,
                                                    test_size=0.3, 
                                                    random_state=42)
# END SPLIT


# EVALUATE MODEL
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.svm._classes import LinearSVC

ch2 = SelectKBest(chi2, k=1000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

clf = MultinomialNB()
clf = LinearSVC()
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy: %0.3f" % metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, target_names=pd.unique(data['genre'])))
# END EVALUATE