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
from sklearn.metrics import confusion_matrix, classification_report, precision_score

# %matplotlib inline
datafile="../input/islr_data/Default.csv"
data = pd.read_csv(datafile,index_col=0)

# preprocessing
#data = pd.concat([data,pd.get_dummies(data[['default','student']])],axis=1)
# http://stackoverflow.com/questions/40901770/is-there-a-simple-way-to-change-a-column-of-yes-no-to-1-0-in-a-pandas-dataframe
#data.default = data.default.factorize()[0]
#data.student = data.student.factorize()[0]
data.default.replace(('Yes','No'),(1,0),inplace=True)
data.student.replace(('Yes','No'),(1,0),inplace=True)
print(data.head())


# Do not forget to set a random seed...
seed = 42
np.random.seed(seed)

print("copypasta to reproduce graph from book")
# technique for printing graph as in the book
# caveat: only for one predictor
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
5. In Chapter 4, we used logistic regression to predict the probability of default using income and balance on the Default data set.
We will now estimate the test error of this logistic regression model using the validation set approach.
Do not forget to set a random seed before beginning your analysis.
''')
print('''
(a) Fit a multiple logistic regression model that uses 'income' and 'balance' to predict the probability of 'default', using only the observations.
''')
if(1):
  predictors = ['income','balance']
  responsecls = 'default'
  X_full = data[:][predictors]
  y_full = data[:][responsecls]

  regr = linear_model.LogisticRegression()
  pred_full = regr.fit(X_full,y_full).predict(X_full)
  print("score:")
  print(regr.score(X_full,y_full))
  print("confusion matrix for held out data")
  print(confusion_matrix(y_full, pred_full))
  print("overall fraction of correct predictions for the held out data")
  print(classification_report(y_full,pred_full,digits=3))

else:
  print("-I-: Skipping...")

print('''
(b) Using the validation set approach, estimate the test error of this model.
In order to do this, you must perform the following steps:
''')
if(1):
  predictors = ['income','balance']
  responsecls = 'default'
  print('''
  (c) Repeat the process in (b) three times, using three different splits of the observations into a training set and a validation set.
  Comment on the results obtained.
  ''')
  # loop through values of test_size
  test_sizes = [0.2,0.3,0.5]
  for testnum, testsize in enumerate(test_sizes):
    print("i. Split the sample set into a training set and a validation set.")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data[predictors],data[responsecls], test_size=testsize)
    # or do this after generation:  y_test.reset_index(level=int,drop=True)
    print("ii. Fit a multiple logistic regression model using only the training observations.")
    model = make_pipeline(linear_model.LinearRegression())
    model.fit(X_train, y_train)
    print("score:" , model.score(X_test,y_test))
    print('''
    iii. Obtain a prediction of default status for each individual in the validation set by computing the posterior probability of default for that individual, and classifying the individual to the default category if the posterior probability equals 0.5.")
    i.e. convert >0.5 to 1 for default, all other to 0
    ''')
    y_pred = model.predict(X_test)
    # convert to series, indexed based on y_test
    probs = pd.Series(y_pred,index=y_test.index).round()
    # convert >0.5 to 1 based on y_test
    # TODO: verify the threshold, use this instead: probs > 0.5
    probs.round(decimals=0)
    # get rid of negatives
    probs = probs.astype(int)

    print("iv. Compute the validation set error, which is the fraction of the observations in the validation set that are misclassified.")
    matching = (y_test == probs)
    # true / total , where 'True' that the arrays were equal
    ratio_t = matching[matching == True].count() / matching.count()
    ratio_f = matching[matching == False].count() / matching.count()
    print("ratio_t: ", ratio_t)
    print("~ratio_t:", ratio_f)
    print("verify: ratio_t + ~ratio_t:", ratio_t + ratio_f)
else:
  print("-I-: Skipping...")
print('''
(d) Now consider a logistic regression model that predicts the probability of default using income, balance, and a dummy variable for student.
Estimate the test error for this model using the validation set approach.
Comment on whether or not including a dummy variable for student leads to a reduction in the test error rate.
''')
