#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:44:34 2018

@author: chloedinh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from itertools import chain, combinations

#create every single combination of features from the list.
#Taken from stackoverflow solution:https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
def powerset(iterable):
    s = list(set(iterable))
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

df = pd.read_csv('pictures-train.tsv', header=0, delimiter='\t')

#change negative values to nans
df[df['viewed'] < 0] = np.nan
df[df['n_comments'] < 0] = np.nan

#drop all nans
df.dropna(inplace=True)

#Eliminate missing dates and months
df = df[-df['takenon'].str.contains(r'-00')]
df = df[-df['votedon'].str.contains(r'-00')]

#Change columns to date-time format
df['takenon'] = pd.to_datetime(df['takenon'])
df['votedon'] = pd.to_datetime(df['votedon'])

#group by author to see how many picture they uploaded and a proxy for their tenure
author = df.groupby(['author_id'])['etitle'].count()

author_tenure = df.groupby(['author_id'])['takenon'].agg(['min','max'])
author_tenure['tenure'] = author_tenure[max] - author_tenure[min]

#map author tenure to original df
df = df.join(author_tenure,on='author_id',how='left')
df = df.join(author,on='author_id',how='left',rsuffix='_author_count')
#convert tenure column to integer
df['tenure'] = df['tenure'].astype(pd.Timedelta).apply(lambda l: l.days)

#drop min and max column
df.drop(['min','max'], axis=1, inplace=True)

#tenure values are negative, turn into positive
df['tenure'] = df['tenure']*-1

#models
lr = LinearRegression()
logit = LogisticRegression()
ridge = Ridge(50)
rfr = RandomForestRegressor()
rfc = RandomForestClassifier()
gnb = GaussianNB()


#split data into training and testing sets
columns = ['etitle','region','takenon','votedon','viewed','n_comments','tenure','etitle_author_count']
feature_list = list(powerset(columns))
df['votedon'] = df['votedon'].astype('int64')/8.64e+7
df['takenon'] = df['takenon'].astype('int64')/8.64e+7

for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X,df['votes'],test_size=0.3, random_state=42)
    print(feature_set_index)
    
    #Linear Regression
    lr.fit(X_train, y_train)
    if lr.score(X_test, y_test) > .60:
        print("LinearRegression")
        print(pd.Series(lr.coef_, index=index))
        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    
    #Ridge Regression
    ridge.fit(X_train, y_train)
    if lr.score(X_test, y_test) > .60:
        print("Ridge")
        print(pd.Series(ridge.coef_, index=index))
        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    
    #Random Forest Regressor
    rfr.fit(X_train, y_train)
    if rfr.score(X_test, y_test) > .60:
        print("RandomForestRegressor")
        print(pd.Series(rfr.feature_importances_, index=index))
        print(rfr.score(X_train, y_train), rfr.score(X_test, y_test))
    
#try logvote
#df without vote=0 to cal log
df = df[df['votes'] != 0]
df['logvotes'] = np.log(df['votes'])
print('Log Votes')
for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X,df['logvotes'],test_size=0.3, random_state=42)
    print('\n')
    print(feature_set_index)
    
    #Linear Regression
    lr.fit(X_train, y_train)
    if lr.score(X_test, y_test) > .60:
        print('Linear Regression')
        print(pd.Series(lr.coef_, index=index))
        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
        
    #Ridge Rregression
    ridge.fit(X_train, y_train)
    if lr.score(X_test, y_test) > .60:
        print("Ridge")
        print(pd.Series(ridge.coef_, index=index))
        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    
    #Random Forest Regressor
    rfr.fit(X_train, y_train)
    if rfr.score(X_test, y_test) > .60:
        print("RandomForestRegressor")
        print(pd.Series(rfr.feature_importances_, index=index))
        print(rfr.score(X_train, y_train), rfr.score(X_test, y_test))


#todo: Prep for classifiers

#print("LogisticRegression")
#logit.fit(X_train, y_train1)
#print(logit.score(X_train, y_train1), logit.score(X_test, y_test1))
#print(pd.DataFrame(confusion_matrix(y_test1, logit.predict(X_test)),
#                   index=cats, columns=cats))
#
#print("GaussianNB")
#gnb.fit(X_train, y_train1)
#print(gnb.score(X_train, y_train1), gnb.score(X_test, y_test1))
#print(pd.DataFrame(confusion_matrix(y_test1, gnb.predict(X_test)),
#                   index=cats, columns=cats))
#
#print("RandomForestClassifier")
#rfc.fit(X_train, y_train1)
#print(pd.Series(rfc.feature_importances_, index=features))
#print(rfc.score(X_train, y_train1), rfc.score(X_test, y_test1))
