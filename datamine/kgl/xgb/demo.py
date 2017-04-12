import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 5,5
# get_ipython().magic('history -t -f demo.py')
showplt=1 # enable matplotlib calls
dataset="amzn"
dataset="demo"
demodir = "../repo_xgb/demo/data/"
file_train = demodir + "agaricus.txt.train"
file_test  = demodir + "agaricus.txt.test"
if(dataset == "amzn"):
  # TODO: expected f312143, f312141, f312138, f312140, f312145, f312144, f312142, f312139, f312153, f312147, f312149, f312137, f312146, f312150, f312152, f312148, f312151 in input data
  demodir = "../input/kglamzn/"
  file_train = demodir + "train.csv"
  file_test = demodir + "test.csv"
dtrain = xgb.DMatrix(file_train)
dtest = xgb.DMatrix(file_test)
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic',
    'eval_metric': ['auc'],
    }
if(dataset == "amzn"):
  param['objective'] = 'reg:logistic'
num_round = 1
bst=xgb.train(param, dtrain, num_round)
if(showplt):
  # feature importance
  xgb.plot_importance(bst)
  # tree
  xgb.plot_tree(bst)
  plt.show()

preds = bst.predict(dtest)
print("shapes: preds %s dtrain (%d,%d) dtest (%d,%d)" % (preds.shape, dtrain.num_col(),dtrain.num_row(), dtest.num_col(),dtest.num_row()))

# TODO: early stopping for boosting rounds
# https://github.com/dmlc/xgboost/blob/master/doc/python/python_intro.md
# TODO:
# TUTORIAL
# https://jessesw.com/XG-Boost/
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


# scoring
# built-in cross_validation
# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py
num_round = 1
print(xgb.cv(param, dtrain, num_boost_round=num_round,nfold=5, metrics={'auc'}, seed=0)) # metrics: 'error',
print(xgb.cv(param, dtest, num_boost_round=num_round,nfold=5, metrics={'auc'}, seed=0)) # metrics: 'error',

# manual method
from sklearn import (metrics, model_selection, linear_model, preprocessing) # cross_validation is moved to model_selection
# compute AUC metric for this CV fold
fpr, tpr, thresholds = metrics.roc_curve(dtest.get_label(), preds)
roc_auc = metrics.auc(fpr, tpr)
print("AUC on dtest.get_label(),preds (final ): %f" % (roc_auc))

# REFERENCES
# https://github.com/dmlc/xgboost/blob/master/doc/python/python_intro.md
# PLOTTING:
# graphviz: http://stackoverflow.com/a/33433735 - ultimately had to add to %PATH%
# size: http://stackoverflow.com/questions/37340474/xgb-plot-tree-font-size-python
# 
