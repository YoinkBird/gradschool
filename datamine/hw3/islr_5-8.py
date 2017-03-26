#load template.py
# %load template.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.misc
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# %matplotlib inline

# save with:
#%history -f session.py
# BEGIN:
samplesize = 10
samplesize = 100
y = np.random.normal(size=samplesize)
x = np.random.normal(size=samplesize)
x.sort()
y = x - 2*x**2 
y = x - 2*x**2 + np.random.normal(size=samplesize)
x = x.reshape(-1,1)
y = y.reshape(-1,1)
#plt.scatter(x,y)
from sklearn import model_selection, cross_validation
from sklearn import linear_model
# http://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out-loo
# http://stackoverflow.com/questions/24890684/leave-one-out-cross-validation
loo = cross_validation.LeaveOneOut(x.shape[0])
loo = cross_validation.LeaveOneOut(7)

# quick test - what am I doing wrong?
colors = ['teal', 'yellowgreen', 'gold', 'red','green']
plt.scatter(x,y)
plt.show()
# x.shape|y.shape Out[83]: (100, 1)
polydegrees = [1,2,3,4,5]
if(1):
  plt.scatter(x,y)
  for count, degree in enumerate(polydegrees[:2]):
      model = make_pipeline(PolynomialFeatures(degree), LinearRegression()) #Ridge())
      if(1):
        model.fit(x[:10],y[:10])
        y_pred = model.predict(x)
        plt.plot(x, y_pred, color=colors[count], marker='.')
      print(model.score(x,y))
  plt.show()
### # see also http://nbviewer.jupyter.org/github/ipython/ipython/blob/1.x/examples/notebooks/Part%203%20-%20Plotting%20with%20Matplotlib.ipynb
### fig = plt.figure()
### for train_i, test_i in loo:
###     plt.scatter(x,y)
###     degree = 3
###     for count, degree in enumerate([1, 2, 3]):
###         model = make_pipeline(PolynomialFeatures(degree), LinearRegression()) #Ridge())
###         #model.fit(x,y)
###         #plt.plot(x,model.predict(x), 'r-')
###         if(1):
###           model.fit(x,y)
###           y_pred = model.predict(x)
###           plt.plot(x, y_pred, color=colors[count], marker='.')
###         if(0):
###           model.fit(x[train_i],y[train_i])
###           y_pred = model.predict(x[train_i])
###           plt.plot(x[train_i], y_pred, color=colors[count], marker='.')
###     plt.show()
if(1):
    for train_i, test_i in loo:
        plt.scatter(x,y)
        #print("%s %s" % (train_i,test_i))
        #print("%s %s" % (x[train_i],x[test_i]))

        # STUB for linear regression
        regr = linear_model.LinearRegression()
        # reference:
        regr.fit(x,y)
        plt.plot(x,regr.predict(x), 'r-')
        # train:
        regr.fit(x[train_i],y[train_i])
        plt.plot(x[train_i],regr.predict(x[train_i]), 'g.')

    plt.show()

#    plt.plot(x.reshape(-1,1), regr.predict(x.reshape(-1,1)))
# TODO: would need to use axes for super-imposing
#plt.scatter(x,y)

# thought: how to generate 'np.random.normal' from 'np.random.multivariate_normal'
# note: random.normal uses loc=0.0 for 'mean' of dist, 'scale=1.0' for std dev
# np.random.normal(100) ==
#  np.random.multivariate_normal([0,0],[[0,1],[1,0]],100)
