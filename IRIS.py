"""
Created on Fri Apr 19 13:02:07 2019

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


from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state = 1)

clf.fit(feature_train , classes_train)

from IPython.display import Image
from sklearn.externals.six import StringIO
from pydotplus import graph_from_dot_data

dot_data = StringIO()
tree.export_graphviz(clf , out_file = dot_data , feature_names = feature_names)
graph = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(clf, all_features_scaled , all_classes , cv= 10)

cv_scores.mean()

decisiontreescore = clf.score(feature_test , classes_test)
print(decisiontreescore)   #1



