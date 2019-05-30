import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import warnings
warnings.simplefilter("ignore")
import lightgbm as lgb

import sys
sys.path.append('../scripts')
from load_data import load_data
df = load_data()
from tools import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
import optuna
from sklearn.model_selection import train_test_split


def objective(trial):
    X = df.drop(['LibraryName','DiseaseState'],axis  = 1)
    y = df['DiseaseState'].values
    X_train,X_valid,y_train,y_valid = train_test_split(X,y)

    learning_rate = trial.suggest_uniform('learning_rate',0.01,0.5)
    max_depth = trial.suggest_int('max_depth',7,20)
    num_leaves = trial.suggest_int('num_leaves',5,300)
    bagging_fraction = trial.suggest_uniform('bagging_fraction',0.01,0.5)
    feature_fraction = trial.suggest_uniform('feature_fraction',0.01,0.5)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf',1,300)
    lambda_l1 = trial.suggest_int('lambda_l1',5,300)
    lambda_l2 = trial.suggest_int('lambda_l2',5,300)

    param_grid = {
    'learning_rate':learning_rate,
    'max_depth':max_depth,
    'num_leaves':num_leaves,
    'bagging_fraction':bagging_fraction,
    'feature_fraction':feature_fraction,
    'min_data_in_leaf':min_data_in_leaf,
    'lambda_l1':lambda_l1,
    'lambda_l2':lambda_l2
    }
    lgbm_params = {
    'objective': 'multiclass',
    'num_class': 20
    }

    model = lgb.LGBMClassifier(**lgbm_params,**param_grid,n_jobs = -1)
    f1_scoring = make_scorer(f1_score, average='micro',  pos_label=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return (1 - f1_score(y_valid,y_pred, average='micro'))


def opt():
    study = optuna.create_study()
    study.optimize(objective,n_trials = 100)

    print(study.best_params)

if __name__ == '__main__':
    opt()

# {'learning_rate': 0.48148558790197915, 'max_depth': 12, 'num_leaves': 126, 'bagging_fraction': 0.07604494795456074, 'feature_fraction': 0.43596480583156505, 'min_data_in_leaf': 137, 'lambda_l1': 5, 'lambda_l2': 39}