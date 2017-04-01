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
print('''
Ch6 intro: If n >> p - that is, if n, the number of
observations, is much larger than p, the number of variables

8. We will now perform cross-validation on a simulated data set.
(a) Generate a simulated data set as follows:
> set . seed (1)
> y= rnorm (100)
> x= rnorm (100)
> y=x -2* x^2+ rnorm (100)
In this data set, what is n and what is p?
Write out the model used to generate the data in equation form.
''')
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
print('''
(b) Create a scatterplot of X against Y . Comment on what you find.
''')
plt.scatter(x,y)
plt.show()
print('''
(c) Set a random seed, and then compute the LOOCV errors that
result from fitting the following four models using least squares:
i.   Y = B_0 + B_1X + err
ii.  Y = B_0 + B_1X + B_2X2 + err
iii. Y = B_0 + B_1X + B_2X2 + B_3X3 + err
iv.  Y = B_0 + B_1X + B_2X2 + B_3X3 + B_4X4 + err
''')
from sklearn import model_selection, cross_validation
from sklearn import linear_model
# http://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out-loo
# http://stackoverflow.com/questions/24890684/leave-one-out-cross-validation
loo = cross_validation.LeaveOneOut(x.shape[0])

colors = ['teal', 'yellowgreen', 'gold', 'red','green']
if(1):
  # x.shape|y.shape Out[83]: (100, 1)
  polydegrees = [1,2,3,4,5]
if(0):
  scores = list()
  regr = linear_model.LinearRegression()
  plt.scatter(x,y)
  plt.show()
  for count, degree in enumerate(polydegrees[:]):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x)
    score = model_selection.cross_val_score(regr, X_poly, y, cv=loo, scoring='neg_mean_squared_error').mean()
    scores.append(score)
    print(score)
  plt.plot(polydegrees,np.array(scores)*-1)
  plt.show()
if(1):
  scores = list()
  plt.scatter(x,y)
  #for count, degree in enumerate(polydegrees[:2]): #limit to 2
  for count, degree in enumerate(polydegrees[:]):
      model = make_pipeline(PolynomialFeatures(degree), LinearRegression()) #Ridge())
      if(1):
        model.fit(x[:10],y[:10])
        y_pred = model.predict(x)
        plt.plot(x, y_pred, color=colors[count], marker='.')
      score = model.score(x,y)
      scores.append(score)
      print(score)
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

print('''
(d) Repeat (c) using another random seed, and report your results.
Are your results the same as what you got in (c)? Why?
''')
print('''
(e) Which of the models in (c) had the smallest LOOCV error? Is
this what you expected? Explain your answer.
''')
print('''
(f) Comment on the statistical significance of the coefficient estimates that
 results from fitting each of the models in (c) using least squares.
Do these results agree with the conclusions drawn based on the cross-validation results?
''')
