import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import (roc_auc_score, auc, accuracy_score, roc_curve, confusion_matrix)
from scipy import interp
from scipy.stats import fisher_exact

import glob
import os

# K, K-fold cross validations
cv_n_splits = 5
# number of iterations for K-fold cross varidation
n_iter = 10

countfiles = glob.glob('count_genus_level/*')
metafiles = glob.glob('metadata2/*')

for count_file_name in countfiles:
    dataset_id = os.path.basename(count_file_name).replace(".count.genus.level.csv","")
    meta_file_name = "metadata2/"+ dataset_id + ".metadata.txt"
    meta = pd.read_csv(meta_file_name,sep='\t')

    meta["LibraryName"] = meta["LibraryName"].astype(object)
    count =  pd.read_csv(count_file_name,index_col = 0)
    count = count.T
    relative = count.apply(lambda x:x/sum(x),axis=1)
    d_m = pd.merge(meta[["DiseaseState","LibraryName"]],relative, left_on='LibraryName', right_index=True)
    d_m = d_m.dropna(subset=["DiseaseState","Blautia"])
    if d_m.shape[0] == 0 :
        continue
    le = LabelEncoder().fit(d_m['DiseaseState'])
    d_m['DiseaseState'] = le.transform(d_m['DiseaseState'])
    cv = StratifiedKFold(n_splits = 5,shuffle = True ,random_state = 0)
    X_train = d_m.drop(["LibraryName","DiseaseState"],axis = 1)
    y_train = d_m[['DiseaseState']].values

    auc_sum = []
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
        print(meta_file_name.split('/')[-1])
        for train_idx, valid_idx in cv.split(X_train, y_train):
            trn_x = X_train.iloc[train_idx, :]
            val_x = X_train.iloc[valid_idx, :]
            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]
            trn_y = np.reshape(trn_y, (-1))
            val_y = np.reshape(val_y, (-1))
            clf.fit(trn_x, trn_y)
            probs = clf.predict_proba(val_x)[:, 1]
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
        print(roc_auc, fisher_p)
        auc_sum.append(roc_auc)
    print(np.mean(auc_sum))
