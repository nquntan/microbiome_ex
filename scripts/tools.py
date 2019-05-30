import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import (roc_auc_score, auc, accuracy_score, roc_curve, confusion_matrix)
from scipy import interp
from scipy.stats import fisher_exact
import lightgbm as lgb


def printAuc(n_iter,cv_n_splits,datasetName,X_train,y_train):
    auc_sum = []
    interported_auc_sum = []
    clf = RandomForestClassifier(n_estimators=1000)
    for n in range(n_iter):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        conf_mat = np.asarray([[0,0],[0,0]])
        y_probs = np.empty_like(y_train, dtype=float)
        y_trues = np.empty_like(y_train)
        y_preds = np.empty_like(y_train)
        cv_count = 0
        cv_counts = np.empty_like(y_train)
        cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True)
        for train_idx, valid_idx in cv.split(X_train, y_train):
            trn_x = X_train.iloc[train_idx, :]
            val_x = X_train.iloc[valid_idx, :]
            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]
            trn_y = np.reshape(trn_y, (-1))
            val_y = np.reshape(val_y, (-1))
            clf.fit(trn_x, trn_y)
            probs = clf.predict_proba(val_x)[:, 1]
            auc_sum.append(roc_auc_score(val_y, probs))
            # Store probability and true Y for later
            y_probs[valid_idx] = probs.reshape(len(probs), 1)
            y_trues[valid_idx] = val_y.reshape(len(val_y), 1)  # literally redundant, but keep it to maintain backward compatibility
            pred = clf.predict(val_x)
            y_preds[valid_idx] = pred.reshape(len(pred), 1)
            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve(val_y, probs)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            # Compute confusion matrix
            conf_mat += confusion_matrix(val_y, pred, labels=[0, 1])
            # Track which fold each sample was tested in
            cv_counts[valid_idx] = cv_count
            cv_count += 1
        mean_tpr /= cv_n_splits
        # interporated AUC
        roc_auc = auc(mean_fpr, mean_tpr)
        # Fisher's exact test
        _, fisher_p = fisher_exact(conf_mat)
        # output AUC value and P-value using Fisher's exact test
        #print(roc_auc, fisher_p)
        interported_auc_sum.append(roc_auc)
    print('{},{},{}'.format(datasetName,np.mean(interported_auc_sum),np.mean(auc_sum)))
    


def train_model(X_train, y_train, params = None, model_type='lgb',model = None,n_iter=10,folds=5):
    oof = np.zeros(len(X_train))
    score = []
    
    for n in range(n_iter):
        conf_mat = np.asarray([[0,0],[0,0]])
        y_probs = np.empty_like(y_train, dtype=float)
        y_trues = np.empty_like(y_train)
        y_preds = np.empty_like(y_train)
        cv_count = 0
        cv_counts = np.empty_like(y_train)
        cv = StratifiedKFold(n_splits=folds, shuffle=True)
        for train_idx, valid_idx in cv.split(X_train, y_train):
            trn_x = X_train.iloc[train_idx, :]
            val_x = X_train.iloc[valid_idx, :]
            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]
            trn_y = np.reshape(trn_y, (-1))
            val_y = np.reshape(val_y, (-1))

            if model_type ==  'lgb':
                lgb_train = lgb.Dataset(trn_x,trn_y)
                lgb_eval = lgb.Dataset(val_x,val_y,reference=lgb_train)
                lgbm_params = {
                    # 多値分類問題
                    'objective': 'multiclass',
                    'metrics' : 'auc'
                }
                model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)
                probs = model.predict(val_x, num_iteration=model.best_iteration)
                
            score.append(f1_score(val_y, probs,average = 'micro'))
        
    print(np.mean(score))
    
