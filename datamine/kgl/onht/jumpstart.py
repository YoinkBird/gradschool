""" Amazon Access Challenge Starter Code

These files provide some starter code using 
the scikit-learn library. It provides some examples on how
to design a simple algorithm, including pre-processing,
training a logistic regression classifier on the data,
assess its performance through cross-validation and some 
pointers on where to go next.

Paul Duan <email@paulduan.com>
"""

## TODO: clear ipython env before running
##  print("resetting ipython namespace to avoid accidents")
##  get_ipython().magic('reset -f')

from __future__ import division

import numpy as np
from sklearn import (metrics, model_selection, linear_model, preprocessing, ensemble)
import xgboost as xgb

SEED = 42  # always use a seed for randomized procedures


def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    datadir = "../input/kglamzn/"
    data = np.loadtxt(open(datadir + "/" + filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open(datadir + "/" + filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


#def main():
if(1):
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """

    # === load data in memory === #
    print("loading data")
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False) # test has no meaningful labels

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    # tuning: enable/disable individual model's tuning section
    tune_lr = 0
    tune_rfc = 0
    tune_xgb = 0

    # LogisticRegression: 2nd best avg score from 10 rounds of CV:StratifiedShuffleSplit_train:test=0.2 , but 'C' is arbitrary.
    lr_params =  {'C':3, 'multi_class':'ovr', 'solver':'liblinear'}
    lr_name = "LogisticRegressionC%d:%s:%s" % (lr_params['C'], lr_params['multi_class'], lr_params['solver'])
    # XGB: first working conglomeration from tutorial and documentation
    xgb_params =  {'max_depth': 2, 'objective': 'binary:logistic', 'silent': 1}
    models_global = {
        'LogisticRegressionC3:ovr:liblinear'  : linear_model.LogisticRegression(**lr_params),
        'RFC' : ensemble.RandomForestClassifier(),
        'XGB' : xgb.XGBClassifier(**xgb_params),
        }
    # Logistic Regression Parameters
    # src: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    if(tune_lr): # set to '1' for "tuning", then disable once optimal params are found
      lr_classes = ['ovr', 'multinomial']
      lr_solvers = ['newton-cg', 'sag', 'lbfgs'] # Multinomial loss or large dataset
      lr_c_strength=lr_params['C']
      for lr_class in lr_classes:
        for lr_solver in lr_solvers:
          name = "LogisticRegressionC%d:%s:%s" % (lr_c_strength, lr_class, lr_solver)
          models_global[name] = linear_model.LogisticRegression(C=lr_c_strength, multi_class=lr_class, solver=lr_solver)
      # ovm:liblinear - can't create in loop because liblinear doesn't work with multinomial
      lr_params =  {'C':3, 'multi_class':'ovr', 'solver':'liblinear'} # default. Small dataset or L1 penalty
      name = "LogisticRegressionC%d:%s:%s" % (lr_params['C'], lr_params['multi_class'], lr_params['solver'])
      models_global[name] = linear_model.LogisticRegression(**lr_params)
      # remove other models during tuning. may change later.
      del(models_global['RFC'])
      del(models_global['XGB'])
    # working set
    models = {}
    models[lr_name] = models_global[lr_name]
    #del(models['LogisticRegression'])
    #del(models['RFC'])
    #del(models['XGB'])
    # === training & metrics === #
    ################################################################################
    # TODO:
    # * generate several cv models to avoid overfitting
    # * group model by target, i.e. access granted,denied and see if one of these is better
    # * use stratified k-fold instead of shuffling: scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # * add code after the loop to do CV using python methods instead of looping
    # * svm with hyperparameter optimisation using GridSearchCV
    # * LogisticRegression: compare performance of LogisticRegression to statsmodels logit
    # * best subset, boot-strapping, etc
    ################################################################################
    # exp2: Experiment with GridSearchCV to find the right test split - result: failed, test split just correlates with AUC score
    preds = {}
    scores = {}
    for i in range(1,10):
      split_ratio=i*0.05
      name = "GridSearchCV.split:%.02f" % split_ratio
      print("model %s running %d CV rounds using %.02f train:test StratifiedShuffleSplit" % (name,1, split_ratio))
      X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X, y, test_size=split_ratio, random_state=0, stratify=y)
      #gridsearchcv
      # src: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
      # TODO: scorer
      parameters = {'C':[0.1,0.5,0.9,1,1.1,1.5,2,3,10]}
      clf = model_selection.GridSearchCV(linear_model.LogisticRegression(), parameters)
      clf.fit(X_train,y_train)
      # train model and make predictions
      tmppreds = clf.predict_proba(X_cv)[:, 1]

      # compute AUC metric for this CV fold
      fpr, tpr, thresholds = metrics.roc_curve(y_cv, tmppreds)
      roc_auc = metrics.auc(fpr, tpr)
      #print("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
      # record score
      scores[name] = roc_auc
    print("-I-: scores")
    for mdl in sorted(scores, key=scores.get, reverse=True):
      print("%-45s Mean AUC: %f" % (mdl, scores[mdl]))
    ################################################################################
    # exp1: average AUC of random shuffle
    preds = {}
    scores = {}
    n = 10  # repeat the CV procedure 10 times to get more precise results
    #n = 1 # for testing
    split_ratio = 0.20 # 0.2 is best for now; 0.3 reduced score by ~0.08 : from 0.864*** to 0.856***
    for name, model in models.items():
      mean_auc = 0.0
      print("model %s running %d CV rounds using %.02f train:test StratifiedShuffleSplit" % (name,n, split_ratio))
      for i in range(n):
          # for each iteration, randomly hold out 20% of the data as CV set
          # wrapper for: next(ShuffleSplit().split(X, y))
          # DOC: stratify: If not None, data is split in a stratified fashion, using this as the class labels.
          # DOC: no empirical benefit to using 'stratify', using anyway though to avoid negative consequences as this is not i.i.d. data
          X_train, X_cv, y_train, y_cv = model_selection.train_test_split(
              X, y, test_size=split_ratio, random_state=i*SEED, stratify=y)

          # if you want to perform feature selection / hyperparameter
          # optimization, this is where you want to do it

          # train model and make predictions
          model.fit(X_train, y_train) 
          tmppreds = model.predict_proba(X_cv)[:, 1]

          # compute AUC metric for this CV fold
          fpr, tpr, thresholds = metrics.roc_curve(y_cv, tmppreds)
          roc_auc = metrics.auc(fpr, tpr)
          #print("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
          mean_auc += roc_auc
      # record mean score
      scores[name] = mean_auc/n
      #print("%-45s Mean AUC: %f" % (name, mean_auc/n))

      # === Predictions === #
      # When making predictions, retrain the model on the whole training set
      model.fit(X, y)
      # Note: won't be able to score this prediction because the test data has useless labels
      preds[name] = model.predict_proba(X_test)[:, 1]
    # "rank" the scores in descending order
    # src: http://stackoverflow.com/a/16773816 # perl wins at this...
    print("-I-: scores")
    for mdl in sorted(scores, key=scores.get, reverse=True):
      print("%-45s Mean AUC: %f" % (mdl, scores[mdl]))
    if(0):
      #filename = input("Enter name for submission file: ")
      for name, pred in preds.items():
        filename="output" + name
        save_results(pred, filename + ".csv")

#if __name__ == '__main__':
#     main()

# ipython never prints errors, this is the only way to know if the script actually finished.
print("-I-: done")

'''
READING:
  fit_transform - shortcut for fit; transform;
  http://stackoverflow.com/a/43296172

dummies etc
http://stackoverflow.com/questions/40336502/want-to-know-the-diff-among-pd-factorize-pd-get-dummies-sklearn-preprocessing?noredirect=1&lq=1

score
http://stackoverflow.com/questions/40336502/want-to-know-the-diff-among-pd-factorize-pd-get-dummies-sklearn-preprocessing?noredirect=1&lq=1

one hot
http://stackoverflow.com/a/17470183


transform
http://scikit-learn.org/stable/data_transforms.html
'''

'''
TMP RESULTS
'''
'''
model LogisticRegression average score from 10 rounds CV using StratifiedShuffleSplit wth 0.2 train:test split
LogisticRegressionC3:ovr:sag                  Mean AUC: 0.864274
LogisticRegressionC3:ovr:liblinear            Mean AUC: 0.864177
LogisticRegressionDefaultC3                   Mean AUC: 0.864177
LogisticRegressionC3:ovr:newton-cg            Mean AUC: 0.864063
LogisticRegressionC3:ovr:lbfgs                Mean AUC: 0.864007
LogisticRegressionC3:multinomial:newton-cg    Mean AUC: 0.861334
LogisticRegressionC3:multinomial:lbfgs        Mean AUC: 0.861107
LogisticRegressionC3:multinomial:sag          Mean AUC: 0.861006

Result: use 2nd best param, liblinear. WHY: sag is best, BUT consistantly not converging
  sklearn\linear_model\sag.py:286: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  "the coef_ did not converge", ConvergenceWarning
Also: all of the multinomial (many-to-many) estimators performed much worse than the ovr (many-to-one) estimaters
Future: test performance of statsmodels logit
'''
