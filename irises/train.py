import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_iris = pd.read_csv('iris.csv')
#wizualizacje danych
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=data_iris)
plt.show()
sns.lineplot(data=data_iris.drop(['species'], axis=1))
plt.show()
new_data_iris = data_iris.drop(['species'], axis=1)
sns.heatmap(new_data_iris.corr(), annot=True)
plt.show()

#przygotwanie danyc do uczenia i metryk
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

x = data_iris.drop(['species'], axis=1)
y = data_iris['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

#KNN alghoritm

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

print(knn.score(x_test, y_test))

#Logistic Regresion
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x, y)

print(lg.score(x_test, y_test))

#SVM alghorimt
from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma='auto')