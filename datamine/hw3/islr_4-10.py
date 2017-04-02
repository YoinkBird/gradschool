# save with:
#%history -f session.py
#load template.py
# %load template.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.misc
#from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.linear_model import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import model_selection, cross_validation

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score

import statsmodels.api as sm
import statsmodels.formula.api as smf

# %matplotlib inline
datafile="../input/islr_data/Weekly.csv"
datafile="../input/islr_data/Smarket.csv"
data = pd.read_csv(datafile,index_col=0, usecols=range(1,10), parse_dates=True)

# preprocessing
#data = pd.concat([data,pd.get_dummies(data[['default','student']])],axis=1)
# lowercase: http://stackoverflow.com/a/38931854
data.columns = data.columns.str.lower()
# convert 'Up' 'Down' to '1' '0'
data.direction = data.direction.factorize()[0]
print(data.head())
print(data.info())


# Do not forget to set a random seed...
seed = 42
np.random.seed(seed)

print("""
This question should be answered using the Weekly data set, which is part of the ISLR package.
This data is similar in nature to the Smarket data from this chapter's lab,
 except that it contains 1089 weekly returns for 21 years, from the beginning of 1990 to the end of 2010.
Note: Weekly is 1990-2010. Smarket is 2001-2005
""")
print("(a) Produce some numerical and graphical summaries of the Weekly data. Do there appear to be any patterns?")
data.hist()
data.corr().plot() # TODO: seaborn
plt.show()
# year, vulume: 0.539006
# plot volume over "index", which loosely corresponds to year:
plt.scatter(data.index,data['volume'])
plt.show()
#plt.plot(data.index,data['year'])
#plt.show()
print("""
(b) Use the full data set to perform a logistic regression with Direction as the response and the five lag variables plus Volume as predictors. 
Use the summary function to print the results.
Do any of the predictors appear to be statistically significant? If so, which ones?
""")
predictors = ['lag1','lag2','lag3','lag4','lag5','volume']
if(1):
  y_full = data['direction']
  x_full = sm.add_constant(data[predictors])
  est = smf.Logit(y_full,x_full).fit()
  print(est.summary())
  lda_full = LinearDiscriminantAnalysis()
  pred_full = lda_full.fit(x_full,y_full).predict(x_full)
  print(
  '''
  (c) Compute the confusion matrix and overall fraction of correct predictions.
  Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.
  ''')
  print(confusion_matrix(y_full, pred_full))
  print(classification_report(y_full,pred_full,digits=3))
else:
  print("-I-: Skipping...")

print('''
(d) Now fit the logistic regression model using a training data period
from 1990 to 2008, with Lag2 as the only predictor.
Compute the confusion matrix and the overall fraction of correct predictions
for the held out data (that is, the data from 2009 and 2010).
Note: Weekly is 1990-2010. Smarket is 2001-2005. => train:2001-2004 hold:2004-2005
''')
print('''
(e) Repeat (d) using LDA.
''')
if(1):
  # split into test,train datasets
  train_lim = '2004'
  test_start  = '2005'
  predictors = ['lag2']
  X_train = data[:'2004'][predictors]
  y_train = data[:'2004']['direction']

  X_test = data[test_start:][predictors]
  y_test = data[test_start:]['direction']

  lda = LinearDiscriminantAnalysis()
  y_pred = lda.fit(X_train,y_train).predict(X_test)
  print("confusion matrix for held out data")
  # maybe? confusion_matrix(y_train, pred).T
  print(confusion_matrix(y_test, y_pred))
  print("overall fraction of correct predictions for the held out data")
  print(classification_report(y_test,y_pred,digits=3))
else:
  print("-I-: Skipping...")
print('''
(f) Repeat (d) using QDA.
''')
print('''
(g) Repeat (d) using KNN with K = 1.
''')
print('''
(h) Which of these methods appears to provide the best results on
this data?  
''')
print('''
(i) Experiment with different combinations of predictors, includ
ing possible transformations and interactions, for each of the
methods. Report the variables, method, and associated confu
sion matrix that appears to provide the best results on the held
out data. Note that you should also experiment with values for
K in the KNN classifier.
''')


## resources
'''
http://blog.yhat.com/posts/logistic-regression-python-rodeo.html
'''
if(0):
  model = make_pipeline(LogisticRegression()) #Ridge())
#    xcolname = 'direction'
#    x = data['xcolname'].values.reshape(-1,1)
  data.head()
  x = data[:'2005'][['lag1','lag2','lag3','lag4','lag5']]
  x.head()
  y = data[:'2005']['direction']
  model.fit(x,y)
  x2 = x.copy()
  x2.sort(axis=0)
  #y = data.default.values#.reshape(-1,1)
  model.fit(x,y)
  y_pred = model.predict(x)
  # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
  y_pred = model.predict_proba(x)
  plt.scatter(x,y,color='teal')
  # y_pred: 	T : array-like, shape = [n_samples, n_classes]
  plt.plot(x, y_pred[:,1],color='lightblue')# color=colors[count], marker='.')
  plt.show()
  print(model.score(x,y))
