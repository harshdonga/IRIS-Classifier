# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:30:46 2019

@author: Harsh
"""

import pandas as pd

dataset = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' ,names = ['Sepal Length' , 'Sepal Width', 'Petal Length' , 'Petal Width', 'IRIS'])
dataset.describe()

dataset.isnull().sum()

dataset['IRIS'].value_counts()

dataset['IRIS'] = dataset['IRIS'].replace(('Iris-setosa' , 'Iris-versicolor' ,'Iris-virginica') , (1, 2 ,3))


all_features = dataset[['Sepal Length' , 'Sepal Width', 'Petal Length' , 'Petal Width']].values

feature_names = ['Sepal Length' , 'Sepal Width', 'Petal Length' , 'Petal Width']

all_classes = dataset['IRIS'].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

from sklearn.model_selection import train_test_split

feature_train , feature_test , classes_train , classes_test = train_test_split(all_features_scaled , all_classes)

#RandomForest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)

clf.fit(feature_train , classes_train)

score = clf.score(feature_test, classes_test)
print(score)       #97.368

from sklearn.model_selection import cross_val_score

cvscore = cross_val_score(clf , all_features_scaled , all_classes , cv=30)
cvscore.mean()   #95.555

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=123)
clf.fit(feature_train , classes_train)

clf.score(feature_test , classes_test)


cvscore = cross_val_score(clf , all_features_scaled , all_classes , cv=30)
cvscore.mean() 


from sklearn import svm

clf = svm.SVC(kernel= 'linear' , random_state = 1234)
clf.fit(feature_train, classes_train)
clf.score(feature_test , classes_test)
