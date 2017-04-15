The Report.
Document everything that you do, explaining as much as you can.
Clearly, there will be some things that you figure out "just work" and that's ok.
But especially given that the winning code is available, you will get more credit the more that you can explain.
Clever ideas that do not come from the winning solution will be rewarded, even if they are
not actually that clever, i.e., even if they do not improve your score.
The point of this project and this report is for you to demonstrate that you have understood
the ideas and tools from this course, and that you have sufficient mastery of them so you can
start playing around and testing things.
In summary: the report should be a summary of your understanding of everything that you
used and that you think was important in obtaining your final score.
If there were things that did not work, and you think you have an idea why, you may include those as well.
Think of this as a coherent digest of your understanding.
Limit the number of total submissions you make to 100.

Pre-Work
-
pre-work: run dataset under LogisticRegression,RandomForestClassifier,XGB
LogisticRegression: simple starter code supporting onehot for amazon dataset
RandomForestClassifier: setup separately based on sklearn doc. Challenge: none
XGB: setup separately based on xgboost doc.
Once functionality of individual APIs verif
unable to get working with amazon dataset using normal API.
Was able to get it working using the sklearn API
Initial Test Strategy:
The LogisticRegression starter code was best basis as it supported correctly loading data and one-hot encoding of the data
This was then refactored to rank scores and run other types of models, i.e. to loop through a dict of models and their parameters
The test-split function was adjusted to use StratifiedShuffleSplit, meant for categorical data, whereas it had been set to initial ShuffleSplit, meant for i.i.d. data, 
This makes the shuffle algorithm try to evenly distribute the different label values throughout the returned data
E.g. this avoids a situation in which the model is trained mainly on positive outcomes but tested on negative outcomes
This did slightly decrease the auc score by about ~0.01, which indicates that the model was previously overfitting.
This is supported by the fact that the train data consists of ~94% positive outcomes. 
Since negative outcomes are in the minority they are less likely to be evenly distributed without the StratifiedShuffleSplit.
Put differently, this helps ensure that the distribution of outcomes in the randomised data subsets more closely approximates the distribution of the entire dataset.
As the next step, RandomForestClassifier and XGB were added to the automation framework.

Format:
Experiment:
-
<name>
purpose:
train/test split:
details:
result:
weakness:
future:
data:

Pre-Experiment
-
purpose: chooose first model to calibrate, establish baseline score
train/test split:
StratifiedShuffleSplit 0.2, 10 rounds, scored using average roc_auc value over all rounds
details:
The default performance of all models was evaluated locally using 10 CV loops and a StratifiedShuffleSplit 20% test ratio (80:20)
Note: A few of the parameters for the models were changed from the default based on recommendations from the API or otherwise.
Out of thise set, LogisticRegression and RandomForestClassifier performed the best.
result:
These were submitted to kaggle, with RandomForestClassifier scoring 0.84649 and LogisticRegression(C=3) scoring 0.88515
The high value of the LogisticRegression was promising and indicated that it could be quickly maximised using cross-validation.
By contrast, the low value of the other models promised a longer tuning period and less ROI.
weakness:
no optimisation, fixed C value, predicted using the last model fitted from the CV loop instead of model with maximum score

Experiment 1
-
exp1: Calibrate LogisticRegression
purpose:
tune parameters for LogisticRegression to maximise score
train/test split:
StratifiedShuffleSplit 0.2, 10 rounds, scored using average roc_auc value over all rounds
details:
Kept C=3, experimented with all classes and solvers available within LogisticRegression to find the best one
result:
All of the multinomial (many-to-many) estimators performed much worse than the ovr (many-to-one) estimators
This may not be surprising since the data has a binary outcome.
The combinations of ovr:sag and ovr:liblinear each performed well.
ovr:sag had the higher average score, but often could not converge,
whereas ovr:liblinear had a slightly lower score (-0.0001) but always converged.
This also happens to be the default solver for LogisticRegression, indicating that it is a good choice for this dataset.
These are the scores for each model; note that 'DefaultC3' means "all params left as default except for C=3"
adjustment:
For this experiment the optimal test-split was left at 0.2 while testing the optimal model-configuration parameters.
As an exploratory measuer, the model was fitted again on a 7:3 train:test dataset using the previously determined ovr:liblinear solver.
This test ratio of 0.3 reduced the mean score by ~0.08 : from 0.864*** to 0.856***
This indicates that the model was no longer learning enough based on the smaller dataset, and the new ratio was not kept.
A smaller ratio of 0.1 was tried, but the the score increase was too high, thus indicating overfitting.
0.2 was kept as the test ratio, especially since many resources do not recommend a lower ratio and the test data supported this guideline.

weakness:
manual tuning, fixed C value, predicted using the last generated model from the CV loop instead of model with maximum score
future:
Test performance of statsmodels logit

data
model LogisticRegression average score from 10 rounds CV using StratifiedShuffleSplit wth 0.2 train:test split
LogisticRegressionC3:ovr:sag                  Mean AUC: 0.864274
LogisticRegressionC3:ovr:liblinear            Mean AUC: 0.864177
LogisticRegressionDefaultC3                   Mean AUC: 0.864177
LogisticRegressionC3:ovr:newton-cg            Mean AUC: 0.864063
LogisticRegressionC3:ovr:lbfgs                Mean AUC: 0.864007
LogisticRegressionC3:multinomial:newton-cg    Mean AUC: 0.861334
LogisticRegressionC3:multinomial:lbfgs        Mean AUC: 0.861107
LogisticRegressionC3:multinomial:sag          Mean AUC: 0.861006
sag convergence error:
  sklearn\linear_model\sag.py:286: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  "the coef_ did not converge", ConvergenceWarning

Experiment 2
-
exp2: Experiment with GridSearchCV to find the right test split and corresponding C-Value
parameters: test splits from 0.05 to 0.45, C-values from 0.1 to 10
purpose: find optimal test split and corresponding C-value
train/test split:
StratifiedShuffleSplit train:test variable, 1 round, GridSearchCV with default of KFold cv=3, scored using metrics.roc_auc_score
details:
This builds off of the simple test-ratio adjustment step at the end of experiment 1.
Instead of using a fixed C-value, this experiment uses GridSearchCV to quickly iterate through many different C-values.
GridSearchCV chooses optimal parameters (e.g. LogisticRegression C-value) by internally running cross-validation on a list of specified candidates. 

This was run iteratively to narrow in on the best C value by adding more and more valeus.
Ultimately the C-values tended to range between 2 and 3, with smaller test-splits correlating to smaller C-values and larger test splits correlating to bigger C-values.
result:
Failed, test split correlates directly with AUC score
This made it clear that the testing methodology was not robust enough.
It should be possible to evaluate scores locally without having to submit them to kaggle.
weakness:
Manually updating the potential C-values and re-running GridSearchCV

Data:
-I-: scores
GridSearchCV.split:0.05                       Mean AUC: 0.882806
GridSearchCV.split:0.10                       Mean AUC: 0.881686
GridSearchCV.split:0.20                       Mean AUC: 0.874130
GridSearchCV.split:0.15                       Mean AUC: 0.871751
GridSearchCV.split:0.25                       Mean AUC: 0.869473
GridSearchCV.split:0.30                       Mean AUC: 0.869451
GridSearchCV.split:0.35                       Mean AUC: 0.859538
GridSearchCV.split:0.40                       Mean AUC: 0.852844
GridSearchCV.split:0.45                       Mean AUC: 0.850657

Experiment 3
-
understanding kaggle and working with future data
purpose:
rearrange test methodology to use additional dataset: validation hold-out, i.e. train:test:validation
train/test split:
StratifiedShuffleSplit with variable ratios
details:
Experiment2 made it clear that relying on the test data from kaggle would not work.
The test data results are hidden, and only two submission results are allowed per day.
This lead to the decision to treat the kaggle train data as the entire set and split it into three sets: train, test, and validation.
A few values for the validation set size were used with a simple GridSearchCV and submitted to kaggle for evaluation.
result:
kaggle scored the two splits almost identically (possibly within margin of error):
   15% hold-out at (0.87662) and 20% hold-out at (0.87653), i.e. only a difference of 0.00009 .
weakness:
Did not test precise range of C-values, did not use wide range of models to determine the hold-out size.
future:
Re-visit holdout set size once all models are trained

Experiment 2b
-
exp2: redo of Experiment with GridSearchCV to find the right test split and corresponding C-Value
parameters: test splits from 0.05 to 0.45, C-values from 0.1 to 10
purpose: find optimal train:test split and corresponding C-value without influence of validation test set
train/test split:
StratifiedShuffleSplit validation set 15% and 20%, train:test variable rate, 1 round, GridSearchCV with explicit KFold cv=10, scored using metrics.roc_auc_score
details:
This re-runs experiment 2a with an additional validation set.
result:
Success, test split has local maximum AUC score
A validation set of 20% consistenly produced a local maximum.
since the local maximum was found for train:test:validation 1:7:2 this value was used for future model fitting.
Note that the low value of 0.1 could lead to overfitting; however the validation set score is not unusual.
At 15%, no local maximum score could be found for the LogisticRegression.
This observation held up at different times in different programming environments.
weakness:
Only one model tested with new methodology, may not be conclusive
Manually updating the potential C-values and re-running GridSearchCV

Data:
train:test Split:
  notice the local max 0.844486 for "CV KFold" at split == 0.10
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

Experiment 4
-
KNN evaluation
purpose: 
train/test split:
StratifiedShuffleSplit train:72% test:8% validation:20% (7:1:2), 1 round, GridSearchCV with default of KFold cv=3, scored using metrics.roc_auc_score
details:
Evaluating KNN on the model due to its ability to cluster features together
n_neighbors of 3,5,10 were narrowed down to 5,6,7 and finalised on 6
result:
KNN consumes a lot of resources, at times up to 3GB and more.
When trying to fit the model using 6 neighbours on the kaggle dataset, the program crashed due to being out of memory.
weakness:
future:
KNN is resource intensive. It may be worth reducing the number of neighbours simply in order to make a prediction.
Alternatively, this can be attempted again on a computer with more memory.

data:
  scores:
  scoring: roc_auc
  split: StratifiedShuffleSplit
  model: KNeighborsClassifier
  split | params               | CV KFold | train    | validation 0.200
  0.10 | {'n_neighbors': 6}   | 0.802513 | 0.795729 | 0.842452 |
the memory error:
  miniconda3\lib\site-packages\scipy\sparse\base.py in _process_toarray_args(self, order, out)
     1007             return out
     1008         else:
  -> 1009             return np.zeros(self.shape, dtype=self.dtype, order=order)
     1010 
     1011     def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):

  MemoryError: 
freeing memory helped a little bit:
  miniconda3\lib\site-packages\scipy\sparse\compressed.py in _mul_sparse_matrix(self, other)
      494                                     maxval=nnz)
      495         indptr = np.asarray(indptr, dtype=idx_dtype)
  --> 496         indices = np.empty(nnz, dtype=idx_dtype)
      497         data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))
      498 

  MemoryError: 


Experiment 5
-
XGB evaluation
exp5: Running XGB
purpose:
Use XGBoost tree classifier and GridSearchCV to achieve higher accuracy
train/test split:
StratifiedShuffleSplit train:72% test:8% validation:20% (7:1:2), 1 round, GridSearchCV with default of KFold cv=3, scored using metrics.roc_auc_score
details:
Boosted trees are typically good at evaluating classification datasets.
The constant "bootstrapping" of the tree helps to find new patterns in the data.
(i.e. the tree is built up a few levels, then used to re-evaluate the data before adding the next level).

XGBoost is a flexible library which provides a large level of fine-grain control over the way the tree is built.
This is a two part experiment.
Part1: run XGB with least amount of parameter optimisation necessary to fit the model
Part2: use GridSearchCV to find the optimal parameters
result:
part1 unoptimised: low local score (~0.74 on validation set), kaggle score of (0.73847)
Of note is that the validation set score is close to the kaggle score
part2 optimised: local score (~0.871 on validation set), kaggle score of (TBD)
weakness:
Manually updating the potential C-values and re-running GridSearchCV

future:
Use RandomizedSearchCV to determine the parameters as this automatically works on a large range of values
XGBoost has many different parameters, which makes it inefficient to manually specify the potential parameters for CV

Additionally, there are guidelines for the order of optimisation.
E.g. increase the learning rate first to get rough results more quickly,
then tune the "tree shape" (depth, weight, other parameters), 
then decrease the learning rate to "fill in" the rough model.

data:
  split| params                                                           | CV KFold | train    | validation 0.200
  0.10 | {'n_estimators': 100, 'max_depth': 2,  'colsample_bytree': 1}    | 0.715338 | 0.729182 | 0.746372 | # default
  0.10 | {'n_estimators': 500, 'max_depth': 10, 'colsample_bytree': 1}    | 0.831286 | 0.829814 | 0.864375 | # GridSearchCV
  0.10 | {'n_estimators': 600, 'max_depth': 15, 'colsample_bytree': 0.75} | 0.840337 | 0.826703 | 0.870705 | # GridSearchCV


todo:
-
split into true, false data set beceause y.mean() is mostly 1's:
Out[35]: 0.94210992096188473
see what is difference between the two sets as this could be part of the decision factor


General Challenges:
-
1. python API not always clear
2. ipython suppresses error messages
3. python on windows difficult to use

Conclusion
-
