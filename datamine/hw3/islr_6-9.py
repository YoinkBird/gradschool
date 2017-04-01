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
datafile="../input/islr_data/College.csv"
#data = pd.read_csv(datafile,index_col=0, usecols=range(1,10), parse_dates=True)
data = pd.read_csv(datafile, index_col=0)

# preprocessing
#data = pd.concat([data,pd.get_dummies(data[['default','student']])],axis=1)
# lowercase: http://stackoverflow.com/a/38931854
data.columns = data.columns.str.lower()
# convert 'Up' 'Down' to '1' '0'
data.private = data.private.factorize()[0]
#print(data.head())
print(data.info())


# Do not forget to set a random seed...
seed = 42
np.random.seed = seed

print("Produce some numerical and graphical summaries of the data. Do there appear to be any patterns?")
data.hist()
if(0): # annoying :-)
  data.corr().plot() # TODO: seaborn
  plt.show()
  # year, vulume: 0.539006
  # plot volume over "index", which loosely corresponds to year:
  # plot interesting data, e.g. whatever has more correlation
  # plt.scatter(data.index,data['<interesting>'])
  # plt.show()
'''
9. In this exercise, we will predict the number of applications received using the other variables in the College data set.
'''
print(
'''
(a) Split the data set into a training set and a test set.
''')
X_train, X_test, y_train, y_test = model_selection.train_test_split(data.drop(['apps'],axis=1),data.apps, test_size=0.2)
print("train X vs y: %s | %s" % (X_train.shape, y_train.shape))
print("test  X vs y: %s | %s" % (X_test.shape, y_test.shape))
print('''
(b) Fit a linear model using least squares on the training set, and report the test error obtained.
''')
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
# test error
from sklearn.metrics import mean_squared_error
from math import sqrt
mse  = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print("mse: %s | rmse root(%s)" % (mse, rmse))
print("manual method:")
# mean squared error - mean_squared_error
print("mse: " , np.mean((y_pred - y_test) ** 2))
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
print("R^2 score: " , reg.score(X_test,y_test))


print('''
(c) Fit a ridge regression model on the training set, with 'lambda' chosen by cross-validation. Report the test error obtained.
''')
print('''
(d) Fit a lasso model on the training set, with 'lamda' chosen by cross-validation. Report the test error obtained, along with the number of non-zero coefficient estimates.
''')
print('''
(e) Fit a PCR model on the training set, with M chosen by cross-validation. Report the test error obtained, along with the value of M selected by cross-validation.
''')
print('''
(f) Fit a PLS model on the training set, with M chosen by cross-validation.
Report the test error obtained, along with the value of M selected by cross-validation.
''')
print('''
(g) Comment on the results obtained. How accurately can we predict the number of college applications received? Is there much difference among the test errors resulting from these five approaches?
''')

