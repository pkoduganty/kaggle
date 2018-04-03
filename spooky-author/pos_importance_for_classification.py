# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:38:59 2018

@author: 160229
"""
import re
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import spacy

from sklearn import naive_bayes as bayes
from sklearn import svm, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#CONFIGs
MOST_WORDS = 100
class_names = ['EAP', 'HPL', 'MWS']

def label(row):
    for c in class_names:
        if row[c]==1:
            return c
    return None

def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=10):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]
    for coef, feat in topn:
        print(classlabel, feat, coef)
    return topn

def preprocess(text):
    re.sub('\s{2,}', ' ', text) # remove additional whitespace
    #re.sub('[^a-zA-Z]','', text) # remove punctuation
    return text

nlp = spacy.load('en_core_web_sm')

def pos_tag(text):
    doc = nlp(text)
    return ' '.join([ token.tag_ for token in doc ])

def lemma(text):
    doc = nlp(text)
    return ' '.join([ token.lemma_ for token in doc])

def dep(text):
    doc = nlp(text)
    return ' '.join([ token.dep_ for token in doc])

#INPUT
'''
train = pd.read_csv('./input/train.csv').fillna('NULL')
test = pd.read_csv('./input/test.csv').fillna('NULL')

train.index = train['id']
x_train = train['text']
x_train.apply(pos_tag)
y_train = train['author']

test.index = test['id']
x_test = test['text']
x_test.apply(pos_tag)


x_train.to_pickle('x_train_pos.pkl')
y_train.to_pickle('y_train_pos.pkl')
x_test.to_pickle('x_test_pos.pkl')
'''

x_train = pd.read_pickle('x_train_pos.pkl')
y_train = pd.read_pickle('y_train_pos.pkl')
x_test = pd.read_pickle('x_test_pos.pkl')

x_all = pd.concat([x_train, x_test])

vectorizer = CountVectorizer(
    analyzer='word',
    ngram_range=(2, 5),
    max_features=500000)
vectorizer.fit(x_all)
word_index={v:k for k,v in vectorizer.vocabulary_.items()}

train_word_features = vectorizer.transform(x_train)

classifier=bayes.MultinomialNB()
#classifier.fit(train_word_features, y_train)

'''
classifier=svm.LinearSVC()
classifier.fit(train_word_features, y_train)


print('Top n=%d for %s classifier' % (MOST_WORDS,'SVC'))
for c in classifier.classes_:
    most_informative_feature_for_class(vectorizer, classifier, c, 50)
'''

skf = StratifiedKFold(y_train, n_folds=5, shuffle=False, random_state=0)
for i, (train, test) in enumerate(skf):
    classifier.fit(train_word_features[train], y_train[train])
    pred = classifier.predict(train_word_features[test])
    score = metrics.accuracy_score(y_train[test], pred)
    print("accuracy:   %0.3f" % score)
    print("classification report:")
    print(metrics.classification_report(y_train[test], pred, target_names=['EAP','HPL','MWS']))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_train[test], pred))
