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

print("Fit a logistic regression model that uses 'income' and 'balance' to predict 'default'")
if(1):
  model = make_pipeline(LogisticRegression()) #Ridge())
  if(1):
    x = data.balance.values.reshape(-1,1)
    x2 = x.copy()
    x2.sort(axis=0)
    y = data.default.values#.reshape(-1,1)
    model.fit(x,y)
    y_pred = model.predict(x2)
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba
    y_pred = model.predict_proba(x2)
    plt.scatter(x,y,color='teal')
    # y_pred: 	T : array-like, shape = [n_samples, n_classes]
    plt.plot(x2, y_pred[:,1],color='lightblue')# color=colors[count], marker='.')
    plt.show()
  print(model.score(x,y))
else:
  print("-I-: Skipping...")

