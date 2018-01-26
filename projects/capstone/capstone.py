#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:52:50 2018

@author: hwang
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
#import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("~/mlndspace/machine-learning/projects/capstone/data/train.csv")

# explore data
# Data volume - 159571 records total in training data
print data.shape

# Success - Display the first record
display(data.head(n=1))

# Inbalanced classes for all the labels other than 'toxic'
print data.describe() # Doesn't seem to have missing data, great

# Class distribution 
print data[data['toxic']==1].shape

# Interesting that there are 931 rows (0.58%) that are not identified as toxic or severe_toxic, but identified as one of the categories. 
# Could consider remove these rows from the entire dataset
data.loc[(data['toxic'] == 0) & (data['severe_toxic'] == 0) & ((data['obscene'] ==1) | (data['threat'] ==1) | (data['insult'] ==1) | (data['identity_hate'] ==1))]

data.loc[data['obscene']+data['threat']+ data['insult'] + data['identity_hate'] > 1]

# end of exploring data

# pre-process data

# train test split
X = data['comment_text']
y = data['toxic']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
X_train1 = v.fit_transform(X_train)
X_test1 = v.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1')
# class_weight={1:0.1, 0:0.9} didn't make log_loss less... why

clf.fit(X_train1, y_train)

pred1 = clf.predict_proba(X_test1)
# predict gives 0 or 1, predict_proba gives probability

from sklearn.metrics import log_loss
print(log_loss(y_test, pred1)) #0.10901593196414858

# so far the best log loss is 1.47, using everything default. By using l1 penalty, it drops down to 1.34
# if using probability as predictions, the log loss is down to 0.11
print(type(X_train1))

# ====Naive Bayes
# download words from nltk, if words are not downloaded
#import nltk
#nltk.download()

from nltk.corpus import words
word_list = words.words()
print(len(word_list))

# deduplicate the word list
word_list_dedup = list(set(word_list))
print(len(word_list_dedup))

from sklearn.feature_extraction.text import CountVectorizer
v2 = CountVectorizer(stop_words='english', vocabulary=word_list_dedup)
X_train2 = v2.fit_transform(X_train) 
X_test2 = v2.transform(X_test)

from sklearn.naive_bayes import BernoulliNB
clf2 = BernoulliNB(alpha=0.01)
clf2.fit(X_train2, y_train)

pred2 = clf2.predict_proba(X_test2)
print(log_loss(y_test, pred2)) #0.9037973046530914

clf.fit(X_train2, y_train)
print(log_loss(y_test, clf.predict_proba(X_test2))) #0.21714974533893502, changing from using NB to Logistic Reg, loss decreases

X_train2 = v.fit_transform(X_train)
X_test2 = v.transform(X_test)

clf2.fit(X_train2, y_train)

pred2 = clf2.predict_proba(X_test2)
print(log_loss(y_test, pred2)) # 2.7078682127984752. loss increases significantly when switched to use tfidf, likely because Bernoulli is suitable for decrete data. 

# tried to use doc2vec, but the pre-trained model doesn't work. 
# status - failed
#https://github.com/jhlau/doc2vec
import gensim.models as g

pre_trained_model='~/mlndspace/machine-learning/projects/capstone/pre_trained_models/enwiki_dbow/doc2vec.bin'
doc2vec_model = g.doc2vec.Doc2Vec.load(pre_trained_model)
# TODO remove unicode
# TODO stemming
# TODO use some NB that takes continuous input
# TODO improvement - bag of n-grams
# TODO idea - floating window of n-words

import spacy

nlp = spacy.load('en')
#doc = nlp(u"Let's go to N.Y.!")
print(len(X_train)) #127656
print(len(X_test)) #31915

# use spacy tokenizer
# status - failed. still need to figure out how sapcy tokenizer works. 
# do not use
# cannot use test file when tokenizing
raw_text = X_train.append(X_test) # don't think need to change the index
print(len(raw_text))
tokenized_text = list()
for text in raw_text:
    tokenized_text.append(nlp(unicode(text, "utf-8"))) # this is wrong, as all texts should be treated using the same tokenizer
print(len(tokenized_text))
for token in tokenized_text[0]:
    print(token.text)

# fit your own word vectors
# status - failed
import gensim
#model = g.Word2Vec(X_train1.toarray(), size=100)
model = g.Word2Vec(gensim.matutils.Sparse2Corpus(X_train1), size=100)
model = g.Word2Vec(X_train1, size=100)

# use glove
# 'open' requires full path
with open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/glove.6B.50d.txt', "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}
    
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
            

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=400, class_weight={0: 2, 1: 1}))])
    
XX = [['I', 'fucking', 'hate', 'black', '!'], ['Today', 'is', 'a', 'good', 'day', '.'],
      ['This', 'is', 'fucking', 'awesome']]
yy = [['toxic'], ['non-toxic'], ['non-toxic']]

print(type(X_train2))
etree_w2v.fit(X_train2, y_train)

XX_test= [['it', 'is', 'fine', '.'], ['you', 'are', 'fucking', 'black', 'loser'], ['This', 'is', 'fucking', 'stupid']]

pp = etree_w2v.predict_proba(X_test)

print(np.sum(pp))
print(np.sum(y_train))
print(np.sum(y_test))

print(pp[:10])
print(log_loss(y_test, pp))

print(nlp(unicode(list(X_train)[0], "utf-8")))

from spacy.tokenizer import Tokenizer
tokenizer = Tokenizer(nlp.vocab)

tokens = tokenizer(u'This is a sentence. bla bla!')
print(tokens)
print(type(tokens))
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
print(tokens)
print(tokens.to_array(POS))
for s in tokens:
    print(type(s))
    print(s.text)

tokens = list()
texts = [u'One document.', u'...', u'Lots of documents']
for doc in tokenizer.pipe(texts, batch_size=50):
    tokens.append(tokenizer(doc))
print(texts)

# test out stanford tokenizer
import corenlp
import os
os.environ["CORENLP_HOME"] = "/Users/hwang/mlndspace/stanford-corenlp-full-2018-01-31"
tt = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."

    
X_train_tokenized = []
cnt = 0
with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
    for text in X_train:
        text_unicode = unicode(text, 'utf-8')
        ann = client.annotate(text_unicode)
        sentence = ann.sentence[0]
        X_train_tokenized.append([token.word for token in sentence.token])
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1

print(len(X_train_tokenized))
print(X_train_tokenized)

tokenized_train_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/train_tokenized.txt', 'w')
for item in X_train_tokenized:
  print>>tokenized_train_file, item

#sentence = ann.sentence[0]
#[token.word for token in sentence.token]
#token = sentence.token[0]
#print(token.lemma)

X_test_tokenized = []
cnt = 0
with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
    for text in X_test:
        text_unicode = unicode(text, 'utf-8')
        ann = client.annotate(text_unicode)
        sentence = ann.sentence[0]
        X_test_tokenized.append([token.word for token in sentence.token])
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1
        
tokenized_test_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/test_tokenized.txt', 'w')
for item in X_test_tokenized:
  print>>tokenized_test_file, item

print(len(list(y_train)))
print(len(X_train_tokenized))

etree_w2v.fit(X_train_tokenized, list(y_train))
pp = etree_w2v.predict_proba(X_test_tokenized)
print(log_loss(y_test, pp))
# etree balanced 0.3584663033439309
# etree balanced_subsample 0.36040312514602596
# etree None 0.3266151046983761
# etree {0:2,1:1} the best 

cnt = 0
for i,l in enumerate(y_test):
    if l == 1:
        cnt += 1
print(cnt)
cnt = 0
for i,l in enumerate(pp):
    if l[1] >= 0.5:
        cnt += 1      
print(cnt)

print(X_test_tokenized[7])



XX_test= [['it', 'is', 'fine', '.'], ['you', 'are', 'fucking', 'black', 'loser'], ['This', 'is', 'fucking', 'awesome']]
print(etree_w2v.predict(XX_test))

import sys, os, re, csv, codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
list_tokenized_train = tokenizer.texts_to_sequences(X_train)

maxlen = 200
X_train_seq = pad_sequences(list_tokenized_train, maxlen=maxlen)

inp = Input(shape=(maxlen, )) 

embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_train_seq,y_train, batch_size=batch_size, epochs=epochs)


list_tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test_seq = pad_sequences(list_tokenized_test, maxlen=maxlen)

pred = model.predict(X_test_seq)

