import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, auc, accuracy_score)
from sklearn.metrics import (roc_auc_score, auc, accuracy_score, roc_curve, confusion_matrix)
from sklearn.metrics import f1_score
from tqdm import tqdm
tqdm.pandas()
import warnings
warnings.simplefilter("ignore")
import lightgbm as lgb

import sys
sys.path.append('../scripts')
from load_data import load_data
df = load_data()

df['DiseaseState'] = labelEncode(df,'DiseaseState')
X_train = df.drop(['LibraryName','DiseaseState'],axis  = 1)
y_train = df['DiseaseState'].values

pram_grid = {
    'learning_rate':[0.01,0.1,0.5],
    'max_depth':[5,6,7,8,9,10],
    'num_leaves':[0,5,10,15,20],
    'baggin_fraction':[0,0.01,0.5,1],
    'feature_fraction':[0,0.01,0.5,1],
    'min_data_in_leaf':[0,5,15,300],
    'lambda_l1':[0,5,15,300],
    'lambda_l2':[0,5,15,300]
}
lgbm_params = {
    # 多値分類問題
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 20
}
lgb_train = lgb.Dataset(trn_x,trn_y)
lgb_eval = lgb.Dataset(val_x,val_y,reference=lgb_train)

gs = GridSearchCV(estimator=lgb.LGBMClassifier(),
                 param_grid = pram_grid,
                 scoring = 'f1',
                 cv = 5,
                 n_jobs = -1)

gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)