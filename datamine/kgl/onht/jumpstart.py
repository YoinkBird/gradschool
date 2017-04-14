""" Amazon Access Challenge Starter Code

original version obtained at:
https://www.kaggle.com/c/amazon-employee-access-challenge/discussion/4797
main leftovers: data loading and onehot encoding
otherwise mostly changed
"""

## TODO: clear ipython env before running
##  print("resetting ipython namespace to avoid accidents")
##  get_ipython().magic('reset -f')

from __future__ import division

import numpy as np
from sklearn import (metrics, model_selection, linear_model, preprocessing, ensemble)
import xgboost as xgb
import pprint as pp
import re

SEED = 42  # always use a seed for randomized procedures

save_files = 0

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
    required_shape = 58921
    if(predictions.shape[0] != required_shape):
      print("-E-: shape should be %d, is %d" % (required_shape, predictions.shape[0]))
      return

    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

# dump hash of predictions to a file
def save_all_results(predictions):
  for name, pred in preds.items():
    filename="output" + name
    filename = re.sub(':', '_', filename)
    save_results(pred, filename + ".csv")
  return


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
    y_kaggle, X_kaggle = load_data('test.csv', use_labels=False) # test has no meaningful labels

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_kaggle)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_kaggle = encoder.transform(X_kaggle)

    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after
    print("START splitting data")

    # allow for train:test:validation split
    # default: no actual "validation" hold-out set:
    #          refer to kaggle as validation set, refer to X,y as the set containing train+test
    validation_size=0.0
    X_train_test = X
    y_train_test = y
    X_validation = X_kaggle
    y_validation = y_kaggle

    # otherwise: set validation_size to >0 to use part of 'X' as a validation hold-out set
    #            and split the remainder into train+test as needed
    # this is because the 'y_kaggle' is not labelled and thus cannot be used for scoring
    # create hold-out validation set by splitting into two sets: "train+test" and "validatoin"
    # strategy:
    # split data from 'train.csv' into three sets
    # "train+test":"validation" 80:20

    validation_size=0.15 # kaggle: marginally best  score # locally: no optimal train/test though
    validation_size=0.20 # kaggle: marginally worse score # locally: allows for optimal train/test to be found
    print("validation hold-out set size: %f" % validation_size)
    if(validation_size != 0):
      print("setting aside %f percent of the train.csv data for validation")
      X_train_test, X_validation, y_train_test, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=0, stratify=y)
    print("manually verify data format by visual inspection:")
    for data in [ X_train_test, X_validation, y_train_test, y_validation ]:
      print("data shape: %s type: %s" % (data.shape,type(data)))
    print("X shape: %s type: %s" % (X.shape,type(X)))
    print("y shape: %s type: %s" % (y.shape,type(y)))
    print("X_train_test shape: %s type: %s" % (X_train_test.shape,type(X_train_test)))
    print("y_train_test shape: %s type: %s" % (y_train_test.shape,type(y_train_test)))
    print("X_validation shape: %s type: %s" % (X_validation.shape,type(X_validation)))
    print("y_validation shape: %s type: %s" % (y_validation.shape,type(y_validation)))

    print("DONE splitting data")
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
    print("# exp3: kfold cross validation with GridSearchCV")
    preds = {}
    scores = {}
    if(1):
      cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
      parameters = {'C':[0.1,0.5,0.9,1,1.1,1.5,1.8,1.9,1.99,2,2.01,2.1,2.3,2.5,3,10]}
      gridsearch_scorer = 'neg_mean_squared_error'
      gridsearch_scorer = 'roc_auc'
      clf_grid = model_selection.GridSearchCV(linear_model.LogisticRegression(), parameters, cv=10, scoring=gridsearch_scorer)
      print("# ideallistic fit on original 'train','test' data set: fit(X,y);predict_proba(X_kaggle)")
      print("finding the traintest:validation split - this one: %.02f" % validation_size)
      # TMP: hijacking to fit on the train/test real quick in order to use kaggle to verify the train/test split
      if(1):
        # testing against entire dataset with NO split to establish some sort of baseline - didn't submit to kaggle yet, want to try with best model
        #clf_grid.fit(X,y)
        # testing against test+train dataset split, submitted to kaggle with vals of 0.15 and 0.20 for validation set size
        clf_grid.fit(X_train_test,y_train_test)
        tmppreds = clf_grid.predict_proba(X_kaggle)[:,1]
        preds["GridSearchCV_kaggleset_%.02f" % validation_size] = tmppreds
        print("GridSearchCV results: %s | %s: %f" % (clf_grid.best_params_, clf_grid.scoring, clf_grid.best_score_))
        print("GridSearchCV best estimator:")
        pp.pprint(clf_grid.best_estimator_)
        #print("GridSearchCV cv_results_:")
        #pp.pprint(clf_grid.cv_results_)
      else:
        print("-I-: Skipping...")
      print("# ideallistic fit on 'train+test','validation' set: fit(X_train_test,y_train_test); predict_proba(X_validation)")
      if(1):
        # testing against smaller validation dataset - don't save this result to 'preds'
        clf_grid.fit(X_train_test,y_train_test)
        tmppreds = clf_grid.predict_proba(X_validation)[:,1]
        print("GridSearchCV results: %s | %s: %f" % (clf_grid.best_params_, clf_grid.scoring, clf_grid.best_score_))
        print("GridSearchCV best estimator:")
        pp.pprint(clf_grid.best_estimator_)
        #print("GridSearchCV cv_results_:")
        #pp.pprint(clf_grid.cv_results_)
      else:
        print("-I-: Skipping...")
    else:
      print("-I-: Skipping...")
    if(save_files):
      for name, pred in preds.items():
        filename="output" + name
        filename = re.sub(':', '_', filename)
        save_results(pred, filename + ".csv")
    ################################################################################
    print("# exp2: Experiment with GridSearchCV+LogisticRegresstion to find the right test split\n\t\t- result: potentially optimal ratio for train:test:validation of 7:1:2")
    preds = {}
    scores = {}
    scores_mse = {}
    # try train/test ratios from 0.05 through 0.45
    split_ratios =  np.arange(5, 50, 5) / 100
    split_ratios = [0.1] # consistently results in local max CV score combined with current validation set
    # replaces: for i in range(1,10): split_ratio=i*0.05
    print("# trying split_ratios: %s" % split_ratios)
    print("scoring: %s" % ("roc_auc"))
    print("split: %s" % "StratifiedShuffleSplit")
    modelname = "LogisticRegression"
    print("model: %s" % (modelname))
    # header:
    # split | params     | roc_auc CV | roc_auc train | roc_auc global-holdout
    # :0.05 | {'C': 2.5} | 0.836104   | 0.884519      | 0.874113
    colwidth_params=20
    print("%-s | %-*s | %-8s | %-8s | %-8s %.03f" % ("split" , colwidth_params, "params" , "CV KFold" , "train" , "validation", validation_size))
    for split_ratio in (split_ratios):
      model = linear_model.LogisticRegression()
      # GridSearchCV with default cv (kfold==3) running on test-split
      name = "StratifiedShuffleSplit:%.02f:GridSearchCV:%s" % (split_ratio, modelname)
      X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X_train_test, y_train_test, test_size=split_ratio, random_state=0, stratify=y_train_test)
      #gridsearchcv
      # src: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
      # TODO: scorer
      parameters = {'C':[0.1,0.5,0.9,1,1.1,1.5,2,3,10]}
      parameters = {'C':[0.1,0.5,0.9,1,1.1,1.5,1.8,1.9,1.99,2,2.01,2.1,2.3,2.5,3,10]}
      clf = model_selection.GridSearchCV(model, parameters, scoring='roc_auc')
      # train model and make predictions
      clf.fit(X_train,y_train)
      # compute AUC metric for this CV fold
      # score for cross-validation
      y_cv_roc = metrics.roc_auc_score(y_cv, clf.predict_proba(X_cv)[:,1])
      # score for hold-out validation
      tmppreds = clf.predict_proba(X_validation)[:, 1]

      # MSE
      tmpmse = metrics.mean_squared_error(y_validation,tmppreds)
      # compute AUC metric for this CV fold
      tmp_roc = metrics.roc_auc_score(y_validation,tmppreds)
      fpr, tpr, thresholds = metrics.roc_curve(y_validation, tmppreds)
      roc_auc = metrics.auc(fpr, tpr)
      print("%-.002f | %-*s | %f | %f | %f" % (split_ratio, colwidth_params, clf.best_params_, clf.best_score_, y_cv_roc, tmp_roc))
      # record score
      scores[name] = roc_auc
      scores_mse[name] = tmpmse
    print("-I-: scores")
    print("%-45s Mean AUC:" % ("model"))
    for mdl in sorted(scores, key=scores.get, reverse=True):
      print("%-45s : %f" % (mdl, scores[mdl]))
    print("%-45s Mean MSE:" % ("model"))
    for mdl in sorted(scores_mse, key=scores.get, reverse=True):
      print("%-45s : %f" % (mdl, scores_mse[mdl]))
    ################################################################################
    print("# exp1: average of random shuffle")
    preds = {}
    scores = {}
    scores_mse = {}
    n = 10  # repeat the CV procedure 10 times to get more precise results
    #n = 1 # for testing
    split_ratio = 0.20 # 0.2 is best for now; 0.3 reduced score by ~0.08 : from 0.864*** to 0.856***
    for name, model in models.items():
      mean_auc = 0.0
      mean_mse = 0.0
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

          # MSE
          tmpmse = metrics.mean_squared_error(y_cv,tmppreds)
          #print("MSE:" , metrics.mean_squared_error(y_cv,tmppreds))
          mean_mse += tmpmse
          # compute AUC metric for this CV fold
          fpr, tpr, thresholds = metrics.roc_curve(y_cv, tmppreds)
          roc_auc = metrics.auc(fpr, tpr)
          #print("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
          mean_auc += roc_auc
      # record mean score
      scores[name] = mean_auc/n
      scores_mse[name] = mean_mse/n
      #print("%-45s Mean AUC: %f" % (name, mean_auc/n))

      # === Predictions === #
      # When making predictions, retrain the model on the whole training set
      model.fit(X, y)
      # Note: won't be able to score this prediction because the test data has useless labels
      preds[name] = model.predict_proba(X_test)[:, 1]
    # "rank" the scores in descending order
    # src: http://stackoverflow.com/a/16773816 # perl wins at this...
    print("-I-: scores")
    print("%-45s Mean AUC:" % ("model"))
    for mdl in sorted(scores, key=scores.get, reverse=True):
      print("%-45s : %f" % (mdl, scores[mdl]))
    print("%-45s Mean MSE:" % ("model"))
    for mdl in sorted(scores_mse, key=scores.get, reverse=True):
      print("%-45s : %f" % (mdl, scores_mse[mdl]))
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
'''
train:test Split:
  notice the local max CV KFold at 0.10
scoring: roc_auc
split: StratifiedShuffleSplit
model: LogisticRegression
split | params     | CV KFold | train    | validation 0.200
0.05 | {'C': 2.5}  | 0.836104 | 0.884519 | 0.874113
0.10 | {'C': 3}    | 0.844486 | 0.836512 | 0.872174
0.15 | {'C': 2.5}  | 0.834175 | 0.835599 | 0.870892
0.20 | {'C': 2.01} | 0.828790 | 0.843344 | 0.868624
0.25 | {'C': 2.5}  | 0.825378 | 0.838450 | 0.865420
0.30 | {'C': 2.3}  | 0.817830 | 0.839385 | 0.861690
0.35 | {'C': 3}    | 0.819672 | 0.837765 | 0.850022
0.40 | {'C': 3}    | 0.813203 | 0.835388 | 0.843198
0.45 | {'C': 3}    | 0.819064 | 0.831954 | 0.839426
'''
