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

# %matplotlib inline
datafile="../input/islr_data/Default.csv"
data = pd.read_csv(datafile,index_col=0)

# preprocessing
#data = pd.concat([data,pd.get_dummies(data[['default','student']])],axis=1)
# http://stackoverflow.com/questions/40901770/is-there-a-simple-way-to-change-a-column-of-yes-no-to-1-0-in-a-pandas-dataframe
data.default.replace(('Yes','No'),(1,0),inplace=True)
data.student.replace(('Yes','No'),(1,0),inplace=True)
print(data.head())


# Do not forget to set a random seed...
seed = 42
np.random.seed(seed)

print('''
5. In Chapter 4, we used logistic regression to predict the probability of default using income and balance on the Default data set.
We will now estimate the test error of this logistic regression model using the validation set approach.
Do not forget to set a random seed before beginning your analysis.
''')
print('''
(a) Fit a multiple logistic regression model that uses 'income' and 'balance' to predict the probability of 'default', using only the observations.
''')
if(1):
  # technique for printing graph as in the book
  if(1):
    model = make_pipeline(LogisticRegression()) #Ridge())
    X_full = data.balance.values.reshape(-1,1)
    X_sorted = X_full.copy()
    X_sorted.sort(axis=0)
    y_full = data.default.values#.reshape(-1,1)
    model.fit(X_full,y_full)
    y_pred = model.predict(X_sorted)
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
    y_pred = model.predict_proba(X_sorted)
    plt.scatter(X_full,y_full,color='teal')
    # y_pred: 	T : array-like, shape = [n_samples, n_classes]
    plt.plot(X_sorted, y_pred[:,1],color='lightblue')# color=colors[count], marker='.')
    plt.show()
    print(model.score(X_full,y_full))
else:
  print("-I-: Skipping...")

print('''
(b) Using the validation set approach, estimate the test error of this model.
In order to do this, you must perform the following steps:
''')
print("i. Split the sample set into a training set and a validation set.")
print("ii. Fit a multiple logistic regression model using only the training observations.")
print("iii. Obtain a prediction of default status for each individual in the validation set by computing the posterior probability of")
print("default for that individual, and classifying the individual to the default category if the posterior probability equals 0.5.")
print("iv. Compute the validation set error, which is the fraction of the observations in the validation set that are misclassified.")
print('''
(c) Repeat the process in (b) three times, using three different splits of the observations into a training set and a validation set.
Comment on the results obtained.
''')
print('''
(d) Now consider a logistic regression model that predicts the probability of default using income, balance, and a dummy variable for student.
Estimate the test error for this model using the validation set approach.
Comment on whether or not including a dummy variable for student leads to a reduction in the test error rate.
''')
