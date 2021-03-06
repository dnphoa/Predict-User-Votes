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
import matplotlib.pyplot as plt

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

#convert datetime values into day of the week
df['votedon_weekday'] = df['votedon'].dt.dayofweek
df['takenon_weekday'] = df['takenon'].dt.dayofweek


#convert day of the week into bool - weekday = 1, weekend = 0
df = df.assign(takenon_weekday_bool = lambda x: x.takenon_weekday < 5)
df = df.assign(votedon_weekday_bool = lambda x: x.votedon_weekday < 5)

#models
lr = LinearRegression()
logit = LogisticRegression()
ridge = Ridge(50)
rfr = RandomForestRegressor()
rfc = RandomForestClassifier()
gnb = GaussianNB()
best = 0
best_info = []

#split data into training and testing sets
#columns = ['etitle','region','takenon','votedon','viewed','n_comments','tenure','etitle_author_count']
columns = ['etitle','region','viewed','n_comments','tenure','etitle_author_count','takenon_weekday_bool','votedon_weekday_bool']
feature_list = list(powerset(columns))
df['votedon'] = df['votedon'].astype('int64')/8.64e+7
df['takenon'] = df['takenon'].astype('int64')/8.64e+7

print('Votes Regressions')
for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X,df['votes'],test_size=0.3, random_state=42)
    print(feature_set_index)
    
    #Linear Regression
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes Regressions', 'Linear Regression', feature_set_index]
#    if lr.score(X_test, y_test) > .75:
#        print("LinearRegression")
#        print(pd.Series(lr.coef_, index=index))
#        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    
    #Ridge Regression
    ridge.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes Regressions','Ridge Regression', feature_set_index]
#    if lr.score(X_test, y_test) > .75:
#        print("Ridge")
#        print(pd.Series(ridge.coef_, index=index))
#        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    
    #Random Forest Regressor
    rfr.fit(X_train, y_train)
    score = rfr.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes Regressions','Random Forest Regressor', feature_set_index]
#    if rfr.score(X_test, y_test) > .75:
#        print("RandomForestRegressor")
#        print(pd.Series(rfr.feature_importances_, index=index))
#        print(rfr.score(X_train, y_train), rfr.score(X_test, y_test))
#Best = 0.4651750177793057
    
    
    
    
    
#try logvote
#df without vote=0 to cal log
print('Log Votes Regression')
df = df[df['votes'] != 0]
df['logvotes'] = np.log(df['votes'])
for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X,df['logvotes'],test_size=0.3, random_state=42)
#    print('\n')
    print(feature_set_index)
    
    #Linear Regression
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes Regression', 'Linear Regression', feature_set_index]
#    if lr.score(X_test, y_test) > .75:
#        print('Linear Regression')
#        print(pd.Series(lr.coef_, index=index))
#        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
        
    #Ridge Rregression
    ridge.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes Regression', 'Ridge Regression', feature_set_index]
#    if lr.score(X_test, y_test) > .75:
#        print("Ridge")
#        print(pd.Series(ridge.coef_, index=index))
#        print(lr.score(X_train, y_train), lr.score(X_test, y_test))
    
    #Random Forest Regressor
    rfr.fit(X_train, y_train)
    score = rfr.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes Regression', 'Random Forest Regressor', feature_set_index]
#    if rfr.score(X_test, y_test) > .75:
#        print("RandomForestRegressor")
#        print(pd.Series(rfr.feature_importances_, index=index))
#        print(rfr.score(X_train, y_train), rfr.score(X_test, y_test))
#best = 0.51335208676310073







#todo: Prep for classifiers - log votes + qcuts
print('Log Votes QCut')
df["votesqcut"] = pd.qcut(df['logvotes'], 4, labels = [1,2,3,4])
for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X, df["votesqcut"], 
                              test_size=0.3, random_state=42)
#    print('\n')
    print(feature_set_index)
    

    #Logistic Regression
    logit.fit(X_train, y_train)
    score = logit.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes QCut', 'Logistic Regression', feature_set_index]
#    if logit.score(X_test, y_test) > 0.75: 
#        print("LogisticRegression")
#        print(logit.score(X_train, y_train), logit.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, logit.predict(X_test)),
#                           index=index, columns=index))

    #GaussianNB
    gnb.fit(X_train, y_train)
    score = gnb.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes QCut', 'GaussianNB', feature_set_index]
#    if gnb.score(X_test, y_test) > 0.75:
#        print("GaussianNB")
#        print(gnb.score(X_train, y_train), gnb.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, gnb.predict(X_test)),
#                           index=index, columns=index))

    #RandomForestClassifier
    rfc.fit(X_train, y_train)
    score = rfc.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes QCut', 'Random Forest Classifier', feature_set_index]
#    if rfc.score(X_test, y_test) > 0.75:
#        print("RandomForestClassifier")
#        print(pd.Series(rfc.feature_importances_, index=index))
#        print(rfc.score(X_train, y_train), rfc.score(X_test, y_test))


#best = above 






#todo: Prep for classifiers - votes + qcuts
print('Votes QCut')
df["votesqcut"] = pd.qcut(df['votes'], 4, labels = [1,2,3,4])
for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X, df["votesqcut"], 
                              test_size=0.3, random_state=42)
#    print('\n')
    print(feature_set_index)
    

    #Logistic Regression
    logit.fit(X_train, y_train)
    score = logit.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes QCut', 'Logistic Regression', feature_set_index]
#    if logit.score(X_test, y_test) > 0.75: 
#        print("LogisticRegression")
#        print(logit.score(X_train, y_train), logit.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, logit.predict(X_test)),
#                           index=index, columns=index))

    #GaussianNB
    gnb.fit(X_train, y_train)
    score = gnb.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes QCut', 'GaussianNB', feature_set_index]
#    if gnb.score(X_test, y_test) > 0.75:
#        print("GaussianNB")
#        print(gnb.score(X_train, y_train), gnb.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, gnb.predict(X_test)),
#                           index=index, columns=index))

    #RandomForestClassifier
    rfc.fit(X_train, y_train)
    score = rfc.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes QCut', 'Random Forest Classifier', feature_set_index]
#    if rfc.score(X_test, y_test) > 0.75:
#        print("RandomForestClassifier")
#        print(pd.Series(rfc.feature_importances_, index=index))
#        print(rfc.score(X_train, y_train), rfc.score(X_test, y_test))
#best = .72930380050929711
#['Votes QCut','Random Forest Classifier',255]






#todo: Prep for classifiers - log votes + pd.cut
print('Log Votes Cut')
df["logvotescut"] = pd.cut(df['logvotes'], [-np.inf,1.37262346,2.74524692,4.57541153,np.inf], labels = [1,2,3,4])
cats = df['votescut'].cat.categories
for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X, df["logvotescut"], 
                              test_size=0.3, random_state=42)
#    print('\n')
    print(feature_set_index)
    

    #Logistic Regression
    logit.fit(X_train, y_train)
    score = logit.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes Cut', 'Logistic Regression', feature_set_index]
#    if logit.score(X_test, y_test) > 0.75: 
#        print("LogisticRegression")
#        print(logit.score(X_train, y_train), logit.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, logit.predict(X_test)),
#                           index=cats, columns=cats))

    #GaussianNB
    gnb.fit(X_train, y_train)
    score = gnb.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes Cut', 'GaussianNB', feature_set_index]
#    if gnb.score(X_test, y_test) > 0.75:
#        print("GaussianNB")
#        print(gnb.score(X_train, y_train), gnb.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, gnb.predict(X_test)),
#                           index=cats, columns=cats))

    #RandomForestClassifier
    rfc.fit(X_train, y_train)
    score = rfc.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Log Votes Cut', 'Random Forest Classifier', feature_set_index]
#    if rfc.score(X_test, y_test) > 0.75:
#        print("RandomForestClassifier")
#        #print(pd.Series(rfc.feature_importances_, index=index))
#        print(rfc.score(X_train, y_train), rfc.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, rfc.predict(X_test)),
#                           index=cats, columns=cats))
#best = .747681733737567867
#['Log Votes Cut','Random Forest Classifier',250]





#todo: Prep for classifiers - votes + pd.cut
print('Votes Cut')
df['votescut'] = pd.cut(df['votes'], [-np.inf, 89.15384615, 177.30769231, 265.46153846, np.inf], labels = [1,2,3,4])
cats = df['votescut'].cat.categories
for features in feature_list[9:]:
    X = pd.get_dummies(df[list(features)])
    index = X.columns
    feature_set_index = feature_list.index(features)
    X_train, X_test, y_train, y_test = tts(X, df["votescut"], 
                              test_size=0.3, random_state=42)
#    print('\n')
    print(feature_set_index)
    

    #Logistic Regression
    logit.fit(X_train, y_train)
    score = logit.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes Cut', 'Logistic Regression', feature_set_index]
#    if logit.score(X_test, y_test) > 0.75: 
#        print("LogisticRegression")
#        print(logit.score(X_train, y_train), logit.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, logit.predict(X_test)),
#                           index=cats, columns=cats))

    #GaussianNB
    gnb.fit(X_train, y_train)
    score = gnb.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes Cut', 'GaussianNB', feature_set_index]
#    if gnb.score(X_test, y_test) > 0.75:
#        print("GaussianNB")
#        print(gnb.score(X_train, y_train), gnb.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, gnb.predict(X_test)),
#                          index=cats, columns=cats))

    #RandomForestClassifier
    rfc.fit(X_train, y_train)
    score = rfc.score(X_test, y_test)
    if score > best:
        best = score
        best_info = ['Votes Cut', 'Random Forest Classifier', feature_set_index]
#    if rfc.score(X_test, y_test) > 0.75:
#        print("RandomForestClassifier")
#        #print(pd.Series(rfc.feature_importances_, index=index))
#        print(rfc.score(X_train, y_train), rfc.score(X_test, y_test))
#        print(pd.DataFrame(confusion_matrix(y_test, rfc.predict(X_test)),
#                          index=cats, columns=cats))

#Best: above
