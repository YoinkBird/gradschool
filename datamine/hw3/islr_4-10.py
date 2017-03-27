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
datafile="../input/islr_data/Weekly.csv"
datafile="../input/islr_data/Smarket.csv"
data = pd.read_csv(datafile,index_col=0)

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

print("Produce some numerical and graphical summaries of the Weekly data. Do there appear to be any patterns?")
data.hist()
data.corr().plot() # TODO: seaborn
plt.show()
# year, vulume: 0.539006
# plot volume over "index", which loosely corresponds to year:
plt.scatter(data.index,data['volume'])
plt.show()
plt.plot(data.index,data['year'])
plt.show()
print("""
Use the full data set to perform a logistic regression with Direction as the response and the five lag variables plus Volume as predictors. 
Use the summary function to print the results.
Do any of the predictors appear to be statistically significant? If so, which ones?
""")
if(0): # not working eyt
  model = make_pipeline(LogisticRegression()) #Ridge())
  if(1):
    xcolname = 'direction'
    x = data['xcolname'].values.reshape(-1,1)
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

