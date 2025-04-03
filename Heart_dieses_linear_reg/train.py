import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

disease_df = pd.read_csv('framingham.csv')
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.dropna(inplace=True, axis=0)

X = np.asarray(disease_df[[
    'age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
Y = np.asarray(disease_df[['TenYearCHD']])

X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)

from sklearn import metrics
print("Dokladnosc", accuracy_score(Y_test, y_pred))

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(Y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm,
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])

plt.figure(figsize = (8, 5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")

plt.show()
print('The details for confusion matrix is =')
print (classification_report(Y_test, y_pred))


