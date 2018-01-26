#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:34:55 2018

@author: hwang
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from time import time

import string

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import dill                           

import corenlp
import os
os.environ["CORENLP_HOME"] = "/Users/hwang/mlndspace/stanford-corenlp-full-2018-01-31"

data = pd.read_csv("~/mlndspace/machine-learning/projects/capstone/data/train.csv")
data.isnull().any()

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = data['comment_text']
y = data[categories]

# split out test data set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# split further to train and validate 
# the final proportions for train, validate, test sets are 60%, 20%, 20%
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=13)

# method to tokenize the raw text
def tokenize_CoreNLP(tokenized, to_tokenize):
    cnt = 0
    with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
        for text in to_tokenize:
            text_unicode = unicode(text, 'utf-8')
            ann = client.annotate(text_unicode)
            sentence = ann.sentence[0]
            tokenized.append([token.word for token in sentence.token])
            if cnt % 1000 == 0:
                print(cnt)
            cnt += 1
    return tokenized

# try to load tokenized input first, if failed, tokenize

# read tokenized input from saved files
import ast

train_tokenized_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/train_tokenized.txt', 'r')
validate_tokenized_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/validate_tokenized.txt', 'r')
test_tokenized_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/test_tokenized.txt', 'r')


def read_tokenized_input_from_file(path):
    lines = path.readlines()
    lines_list = [ast.literal_eval(l) for l in lines]
    path.close()
    return lines_list

#X_train_tokenized = read_tokenized_input_from_file(train_tokenized_file)
#X_validate_tokenized = read_tokenized_input_from_file(validate_tokenized_file)
#X_test_tokenized = read_tokenized_input_from_file(test_tokenized_file)


X_train_tokenized = []
X_train_tokenized = tokenize_CoreNLP(X_train_tokenized, X_train)
tokenized_train_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/train_tokenized.txt', 'w')
for item in X_train_tokenized:
    print>>tokenized_train_file, item

X_validate_tokenized = []
X_validate_tokenized = tokenize_CoreNLP(X_validate_tokenized, X_validate)
tokenized_validate_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/validate_tokenized.txt', 'w')
for item in X_validate_tokenized:
    print>>tokenized_validate_file, item
    
X_test_tokenized = []
X_test_tokenized = tokenize_CoreNLP(X_test_tokenized, X_test)
tokenized_test_file = open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/test_tokenized.txt', 'w')
for item in X_test_tokenized:
    print>>tokenized_test_file, item


session_name = '/Users/hwang/mlndspace/sessions/tokenized_text_session.pkl'
dill.dump_session(session_name)

# and to load the session
#dill.load_session(session_name)

X_train_tokenized_backup = X_train_tokenized
X_validate_tokenized_backup = X_validate_tokenized
X_test_tokenized_backup = X_test_tokenized

X_train_tokenized = pd.Series(string.join(s, " ") for s in X_train_tokenized)
X_validate_tokenized = pd.Series(string.join(s, " ") for s in X_validate_tokenized)
X_test_tokenized = pd.Series(string.join(s, " ") for s in X_test_tokenized)

# a collection of all train_evaluate return values
results = []

def train_evaluate(pipeline, X_train, y_train, X_validate, y_validate, X_test, y_test, categories):
    mean_log_loss = 0.0
    mean_roc = 0.0
    train_time = 0.0
    predict_time = 0.0
    
    mean_log_loss_test = 0.0
    mean_roc_test = 0.0
    
    for c in categories:
        print("training %s " % c)
        y_train_one = y_train[c]
        y_validate_one = y_validate[c]
        y_test_one = y_test[c]
        
        t0 = time()
        pipeline.fit(X_train, y_train_one)
        train_time += time() - t0 # time includes both vectorizing and training
        
        t0 = time()
        pred_validate = pipeline.predict_proba(X_validate)
        predict_time += time() - t0
        
        # metrics are reported on validation set
        l_validate = log_loss(y_validate_one, pred_validate)
        r_validate = roc_auc_score(y_validate_one, pred_validate[:,1])
        print("log loss is %.4f, roc is %.4f" % (l_validate, r_validate))
        mean_log_loss += l_validate
        mean_roc += r_validate
        
        # for model selection purpose, keep track of metrics on test
        pred_test = pipeline.predict_proba(X_test)
        l_test = log_loss(y_test_one, pred_test)
        r_test = roc_auc_score(y_test_one, pred_test[:,1])
        mean_log_loss_test += l_test
        mean_roc_test += r_test
        
    mean_log_loss /= len(categories)
    mean_roc /= len(categories)
    print("VALIDATION METRICS: mean log loss is %.4f, mean roc is %.4f" % (mean_log_loss, mean_roc))
    print("VALIDATION METRICS: training time is %f, prediction time is %f" % (train_time, predict_time))
    
    mean_log_loss_test /= len(categories)
    mean_roc_test /= len(categories)
    print("TEST METRICS: mean log loss is %.4f, mean roc is %.4f" % (mean_log_loss_test, mean_roc_test))
 
    return mean_log_loss, mean_roc, train_time, predict_time, mean_log_loss_test, mean_roc_test

# calculate average log loss and roc, used for CNN and RNN
def get_average_metric(pred, y):
    pred_df = pd.DataFrame(pred)

    cnt = 0
    mean_log_loss = 0.0
    mean_roc = 0.0

    for column in y:
        pred_expand = [[1-p, p] for p in pred_df[cnt]]
        log_loss_one = log_loss(y[column], pred_expand)
        mean_log_loss += log_loss_one
        roc_one = roc_auc_score(y[column], pred_df[cnt])
        mean_roc += roc_one
        cnt += 1
        print("model - %s, log loss - %.4f, roc - %.4f" %(column, log_loss_one, roc_one))
    mean_log_loss /= cnt
    mean_roc /= cnt
    return mean_log_loss, mean_roc

# just a record keeping 
models = ['logistic_regression', 'naive_bayes', 'CNN', 'RNN']
vetorizors = ['tfidf', 'count', 'glove']

# load glove - 50 dimension
with open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/glove.6B.50d.txt', "rb") as lines:
    w2v50 = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}

# load glove - 100 dimension
with open('/Users/hwang/mlndspace/machine-learning/projects/capstone/data/glove.6B.100d.txt', "rb") as lines:
    w2v100 = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}

# define a mean vectorizer to represent a tokenized comment with 
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
            
#etree_w2v = Pipeline([
#    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v50)),
#    ("extra trees", ExtraTreesClassifier(n_estimators=200, class_weight={0: 2, 1: 1}))])

logistic_w2v50 = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v50)), 
        ("logistic regression", LogisticRegression(class_weight='balanced', penalty = 'l1'))]) 

nb = Pipeline([
        ("count vectorizer", CountVectorizer(stop_words='english')), 
        ("multinomial naive bayes", MultinomialNB(alpha = 1.0))])

logistic_count = Pipeline([
        ("count vectorizer", CountVectorizer(stop_words='english')), 
        ("logistic regression", LogisticRegression())
        ])
    
logistic_tfidf = Pipeline([
        ("tfidf vectorizer", TfidfVectorizer()), 
        ("logistic regression", LogisticRegression())
        ])

for p in [nb, logistic_count, logistic_tfidf]:
    print("pipeline:", [name for name, _ in p.steps])
    log_loss_value, roc, train_time, pred_time, log_loss_test, roc_test = train_evaluate(p, X_train_tokenized, y_train, X_validate_tokenized, y_validate, X_test_tokenized, y_test, categories)
    result = [[name for name, _ in p.steps], log_loss_value, roc, train_time, pred_time, log_loss_test, roc_test]
    results.append(result)

session_name = '/Users/hwang/mlndspace/sessions/training_session1.pkl'
dill.dump_session(session_name)

#dill.load_session(session_name)

param_class_weights = ['balanced', {0:0.9, 1:0.1}, {0:0.8, 1:0.2}, {0:0.7, 1:0.3}, {0:0.6, 1:0.4}, {0:0.5, 1:0.5}, {0:0.4, 1:0.6}, {0:0.3, 1:0.7}, {0:0.2, 1:0.8}, {0:0.1, 1:0.9}]
param_penalty = ['l1', 'l2']

min_param = []
min_metric_pairs = []
min_metric = 0
for cw in param_class_weights:
    for p in param_penalty: 
        logistic = Pipeline([
                ("tfidf vectorizer", TfidfVectorizer()), 
                ("logistic regression", LogisticRegression(class_weight=cw, penalty=p))])
        print("training for class weights - %s, penalty - %s" % (str(cw), p))
        log_loss_value, roc, train_time, pred_time, log_loss_test, roc_test = train_evaluate(logistic, X_train_tokenized, y_train, X_validate_tokenized, y_validate, X_test_tokenized, y_test, categories)
        result = ["logistic regression with class weight and penalty " + str(cw) + " " + p, log_loss_value, roc, train_time, pred_time, log_loss_test, roc_test]
        results.append(result)
        if roc > min_metric:
            min_param = [cw, p]
            min_metric_pairs = [log_loss_value, roc]
            min_metric = roc
print("The best auroc achieved for logistic tfidf grid search is, class weight - %s, penalty - %s" % (min_param[0], min_param[1]))

session_name = '/Users/hwang/mlndspace/sessions/training_session2.pkl'
dill.dump_session(session_name)

#dill.load_session(session_name)

# CNN
import sys, os, re, csv, codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
word_index = tokenizer.word_index

list_tokenized_train = tokenizer.texts_to_sequences(X_train)

total_num_words = [len(c) for c in list_tokenized_train]
print("maximum comment length is %.0f" % max(total_num_words))

plt.hist(total_num_words,bins = np.arange(0,1400,10))
plt.show()

maxlen = 200
X_train_seq = pad_sequences(list_tokenized_train, maxlen=maxlen)

# prepare validate sequence
list_tokenized_validate = tokenizer.texts_to_sequences(X_validate)
X_validate_seq = pad_sequences(list_tokenized_validate, maxlen=maxlen)

# prepare test sequence
list_tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test_seq = pad_sequences(list_tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, )) 

embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.4)(x)
x = Dense(6, activation='sigmoid')(x)

model = Model(inputs=inp, outputs=x)

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint 

batch_size = 32
epochs = 4

checkpointer = ModelCheckpoint(filepath='saved_models/weights.cnn.hdf5', 
                               verbose=1, save_best_only=True)

print("traing CNN with self embedding...")

t0 = time()
model.fit(X_train_seq,y_train,
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
train_time = time() - t0
print("VALIDATION METRICS: training time for CNN with self embedding is %f." % train_time)

t0 = time()
pred_validate = model.predict(X_validate_seq)
pred_time = time() - t0
print("VALIDATION METRICS: prediction time for CNN with self embedding is %f." % pred_time)

mean_log_loss, mean_roc = get_average_metric(pred_validate, y_validate)
print("VALIDATION METRICS: log loss for CNN with self embedding is %.4f, roc is %.4f." % (mean_log_loss, mean_roc))

pred_test = model.predict(X_test_seq)
mean_log_loss_test, mean_roc_test = get_average_metric(pred_test, y_test)
print("TEST METRICS: log loss for CNN with self embedding is %.4f, roc is %.4f." % (mean_log_loss_test, mean_roc_test))

result = ["CNN with self embedding", mean_log_loss, mean_roc, train_time, pred_time, mean_log_loss_test, mean_roc_test]
results.append(result)

session_name = '/Users/hwang/mlndspace/sessions/training_session3.pkl'
dill.dump_session(session_name)

#dill.load_session(session_name)

# CNN with GloVe
EMBEDDING_DIM = 50

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = w2v50.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False) # trainable set to False to avoid updating pre-trained embedding

embedded_sequences = embedding_layer(inp)
x = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu')(embedded_sequences)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.4)(x)
x = Dense(6, activation='sigmoid')(x)

model = Model(inputs=inp, outputs=x)

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 4

checkpointer = ModelCheckpoint(filepath='saved_models/weights.cnn.hdf5', 
                               verbose=1, save_best_only=True)

print("traing CNN with GloVe...")

t0 = time()
model.fit(X_train_seq,y_train,
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
print("VALIDATION METRICS: training time for CNN with GloVe is %f." % train_time)

t0 = time()
pred_validate = model.predict(X_validate_seq)
pred_time = time() - t0
print("VALIDATION METRICS: prediction time for CNN with GloVe is %f." % pred_time)

mean_log_loss, mean_roc = get_average_metric(pred_validate, y_validate)
print("VALIDATION METRICS: log loss for CNN with GloVe is %.4f, roc is %.4f." % (mean_log_loss, mean_roc))

pred_test = model.predict(X_test_seq)
mean_log_loss_test, mean_roc_test = get_average_metric(pred_test, y_test)
print("TEST METRICS: log loss for CNN with GloVe is %.4f, roc is %.4f." % (mean_log_loss_test, mean_roc_test))

result = ["CNN with GloVe", mean_log_loss, mean_roc, train_time, pred_time, mean_log_loss_test, mean_roc_test]
results.append(result)

session_name = '/Users/hwang/mlndspace/sessions/training_session4.pkl'
dill.dump_session(session_name)

# RNN
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.summary()
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 4

print("traing RNN with self embedding...")

t0 = time()
model.fit(X_train_seq, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate_seq, y_validate))
train_time = time() - t0
print("VALIDATION METRICS: training time for RNN with self embedding is %f." % train_time)

t0 = time()
pred_validate = model.predict(X_validate_seq)
pred_time = time() - t0
print("VALIDATION METRICS: prediction time for RNN with self embedding is %f." % pred_time)

mean_log_loss, mean_roc = get_average_metric(pred_validate, y_validate)
print("VALIDATION METRICS: log loss for RNN with self embedding is %.4f, roc is %.4f." % (mean_log_loss, mean_roc))

pred_test = model.predict(X_test_seq)
mean_log_loss_test, mean_roc_test = get_average_metric(pred_test, y_test)
print("TEST METRICS: log loss for RNN with self embedding is %.4f, roc is %.4f." % (mean_log_loss_test, mean_roc_test))

result = ["RNN with self embedding", mean_log_loss, mean_roc, train_time, pred_time, mean_log_loss_test, mean_roc_test]
results.append(result)

session_name = '/Users/hwang/mlndspace/sessions/training_session5.pkl'
dill.dump_session(session_name)


# RNN with GloVe
x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.summary()
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 4

print("traing RNN with GloVe...")

t0 = time()
model.fit(X_train_seq,y_train, batch_size=batch_size, epochs=epochs)
print("VALIDATION METRICS: training time for RNN with GloVe is %f." % train_time)

t0 = time()
pred_validate = model.predict(X_validate_seq)
pred_time = time() - t0
print("VALIDATION METRICS: prediction time for RNN with GloVe is %f." % pred_time)

mean_log_loss, mean_roc = get_average_metric(pred_validate, y_validate)
print("VALIDATION METRICS: log loss for RNN with GloVe is %.4f, roc is %.4f." % (mean_log_loss, mean_roc))

pred_test = model.predict(X_test_seq)
mean_log_loss_test, mean_roc_test = get_average_metric(pred_test, y_test)
print("TEST METRICS: log loss for RNN with GloVe is %.4f, roc is %.4f." % (mean_log_loss_test, mean_roc_test))
result = ["RNN with GloVe", mean_log_loss, mean_roc, train_time, pred_time, mean_log_loss_test, mean_roc_test]
results.append(result)

session_name = '/Users/hwang/mlndspace/sessions/training_session6.pkl'
dill.dump_session(session_name)

print("DONE")
