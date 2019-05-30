import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, auc, accuracy_score, roc_curve, confusion_matrix)
from scipy import interp
import warnings
warnings.simplefilter("ignore")
import lightgbm as lgb
from scipy import stats
import codecs
import logging
from sklearn import svm

import sys
sys.path.append('../scripts')
from load_data import load_data

random_state = 0

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    

def train_model(model_name):
    if model_name == 'lgb':
        param = {"min_data_in_leaf":10, 
        # 'min_child_samples':15, 
        'objective':'binary' }
        clf = lgb.LGBMClassifier(**param)
    elif model_name == 'rf':
        clf = RandomForestClassifier(
            n_estimators = 1000,
            # max_depth = 20
            )

    elif model_name == 'svm':
        clf = svm.SVC(probability=True)

    return clf

def cv_and_roc(model_name,X, Y, folds=5, n_iter=10, random_state=None):
    auc_score = []
    clf = train_model(model_name)
    for n in range(n_iter):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0,1,100)
        cv_count = 0
        cv_counts = np.empty_like(Y)
        cv = StratifiedKFold(n_splits = folds, shuffle = True)
        clf = train_model(model_name)
        for train_idx, valid_idx in cv.split(X, Y):
            trn_x, trn_y = X.iloc[train_idx, :], Y[train_idx]
            val_x, val_y = X.iloc[valid_idx, :], Y[valid_idx]
            trn_y = np.reshape(trn_y, (-1))
            val_y = np.reshape(val_y, (-1))
            
            probas = clf.fit(trn_x,trn_y).predict_proba(val_x)[:,1]
            fpr, tpr, thresholds = roc_curve(val_y, probas)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            cv_counts[valid_idx] = cv_count
            cv_count +=1
        mean_tpr /= folds
        roc_auc = auc(mean_fpr, mean_tpr)
        auc_score.append(roc_auc)
    return np.mean(auc_score)

def leave_one_test_roc(X_train, y_train, X_test, y_test, model_name):
    clf = train_model(model_name)
    clf = clf.fit(X_train,y_train)
    probs = clf.predict_proba(X_test)[:,1]

    return roc_auc_score(y_test,probs)  


def plot_compare_auc(res,label,model_name):
    plt.figure()
    grid = sns.FacetGrid(res,hue = 'datasetname', size  = 10)
    grid.map(plt.scatter, 'x','y')
    grid.add_legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot([0, 1],[0.5, 0.5], "gray", linestyle='dashed')
    plt.plot([0.5, 0.5],[0, 1], "gray", linestyle='dashed')
    plt.plot([0, 1],[0, 1], "gray", linestyle='dashed')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('AUC, single-dataset classifier', fontsize=16)
    plt.ylabel('AUC, healty vs disease classifier', fontsize=16)
    plt.title('rf leave one {} out'.format(label), fontsize=22)
    plt.savefig('../output/leave-one-{}-{}.png'.format(label,model_name))
    plt.clf()
    plt.close()


def t_test_of_auc(res,label):
    ans = stats.ttest_rel(res['x'], res['y'])
    with codecs.open('../output/t_test.txt', 'a', 'utf-8') as f:
        f.write('{}:{}'.format(label,ans))
        f.write('\n')


if __name__ == '__main__':
    logger = get_logger()
    df = load_data()

    ######腸内細菌の変化率を特徴量として追加
    df_h = df[df['DiseaseState'] == 'H']
    df_h = df_h.drop(['Age', 'BMI', 'Sex'], axis = 1)
    s = df_h.describe()
    s = s.T['mean']
    df = df.drop(['Sex','Age','BMI'],axis = 1)
    for i in range(5, len(df.columns)):
        df[df.columns[i] + '_changerate'] = df[df.columns[i]] - s[i - 5]
    #######################################################################
    dataset_id = pd.read_csv('../index.txt')
    train_model_list  = ['rf']

    for model_name in train_model_list:

        logger.info('start model name = {}'.format(model_name))
        dataset_res = pd.DataFrame(index=[], columns=['x','y','datasetname'] )
        disease_res = pd.DataFrame(index=[], columns=['x','y','datasetname'] )

        for i in range(dataset_id.shape[0]):
            df_single_dataset = df.query('DiseaseState == "H" or DiseaseState == "{}"'.format(dataset_id['DiseaseState'][i]))
            df_single_dataset = df_single_dataset.query('LibraryName.str.contains("{}")'.format(dataset_id['id'][i]), engine='python')
            
            # df_single_dataset['DiseaseState']= LabelEncoder().fit_transform(df_single_dataset['DiseaseState'])
            X = df_single_dataset.drop(["LibraryName","DiseaseState"],axis = 1)
            y = df_single_dataset[['DiseaseState']].values
            y = np.where(y == 'H', 0, 1)
            single_auc_score = cv_and_roc(model_name,X,y)

            # あとでhalty diseaseに変換するためもう一回呼び出す(上でlabelencodeどうにかする)
            # df_single_dataset = df.query('DiseaseState == "H" or DiseaseState == "{}"'.format(dataset_id['DiseaseState'][i]))
            # df_single_dataset = df_single_dataset.query('LibraryName.str.contains("{}")'.format(dataset_id['id'][i]), engine='python')  

            # leave one dataset
            if df_single_dataset['DiseaseState'].shape[0] > 20:              
                df_leave_dataset = df[~df["LibraryName"].str.contains(dataset_id['id'][i])]
                dataset_X_train = df_leave_dataset.drop(["LibraryName","DiseaseState"],axis = 1)
                dataset_y_train = df_leave_dataset[['DiseaseState']].values
                dataset_y_train = np.where(dataset_y_train == 'H',0,1)
                dataset_X_test = df_single_dataset.drop(["LibraryName","DiseaseState"],axis = 1)
                dataset_y_test= df_single_dataset[['DiseaseState']].values
                dataset_y_test = np.where(dataset_y_test == 'H',0,1)
                dataset_auc_score = leave_one_test_roc(dataset_X_train, dataset_y_train, dataset_X_test, dataset_y_test, model_name)
                
                record = pd.Series([single_auc_score, dataset_auc_score, dataset_id['id'][i]], index=dataset_res.columns) 
                print(single_auc_score, dataset_auc_score, dataset_id['id'][i])
                dataset_res = dataset_res.append(record, ignore_index = True)

            # leave one disease out
            if df_single_dataset['DiseaseState'].shape[0] > 20:
                df_leave_disease = df[df['DiseaseState'] != dataset_id['DiseaseState'][i]]
                df_leave_disease = df_leave_disease[~((df_leave_disease['LibraryName'].str.contains(dataset_id['id'][i])) & (df_leave_disease['DiseaseState'] == 'H'))]

                disease_X_train = df_leave_disease.drop(["LibraryName","DiseaseState"],axis = 1)
                disease_y_train = df_leave_disease[['DiseaseState']].values
                disease_y_train = np.where(disease_y_train == 'H',0,1)
                disease_X_test = df_single_dataset.drop(["LibraryName","DiseaseState"],axis = 1)
                disease_y_test= df_single_dataset[['DiseaseState']].values
                disease_y_test = np.where(disease_y_test == 'H',0,1)
                disease_auc_score = leave_one_test_roc(disease_X_train, disease_y_train, disease_X_test, disease_y_test,model_name)

                record = pd.Series([single_auc_score, disease_auc_score, dataset_id['id'][i]], index=disease_res.columns) 
                disease_res = disease_res.append(record, ignore_index = True)


        #plot leave one dataset and t-test
        print(dataset_res)
        plot_compare_auc(dataset_res,'dataset',model_name)
        t_test_of_auc(dataset_res,'leave one dataset of {}'.format(model_name))


        #plot leave one disease and t-test
        print(disease_res)
        plot_compare_auc(disease_res,'disease',model_name)
        t_test_of_auc(disease_res,'leave one disease of {}'.format(model_name))
