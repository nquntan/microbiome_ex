import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, auc, accuracy_score)
from load_data import load_data
from sklearn.metrics import (roc_auc_score, auc, accuracy_score, roc_curve, confusion_matrix)
from sklearn.metrics import f1_score
from tqdm import tqdm
tqdm.pandas()
import warnings
warnings.simplefilter("ignore")
from sklearn import svm

from healty_vs_disease import cv_and_roc, leave_one_test_roc, train_model

var_name = lambda val : [k for k, v in globals().items() if id(v) == id(val)]

def plot_compare_auc(res,label,dataset,dir):
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
    # plt.xlabel('AUC, single-dataset classifier', fontsize=16)
    # plt.ylabel('AUC, healty vs disease classifier', fontsize=16)
    plt.title('rf leave one {} out'.format(label), fontsize=22)
    plt.savefig('../output/healty_disease/{}/{}-of-leave-one-{}.png'.format(dir,dataset,label))
    plt.clf()
    plt.close()



if __name__ == '__main__':

    df = load_data()
    df['DiseaseState'] = df['DiseaseState'].astype('category')
    # split data 
    dataset = []
    print('all dataset size:{}'.format(df.shape[0]))
    Age = df.drop(["Sex","BMI"],axis = 1).dropna()
    Sex = df.drop(["Age","BMI"],axis = 1).dropna()
    BMI = df.drop(["Age","Sex"],axis = 1).dropna()
    print('Age:{}'.format(Age.shape[0]))
    print('Sex:{}'.format(Sex.shape[0]))
    print('BMI:{}'.format(BMI.shape[0]))
        
    Age_Sex = df.drop(["BMI"],axis = 1).dropna()
    Age_BMI = df.drop(["Sex"],axis = 1).dropna()
    Sex_BMI = df.drop(["Age"],axis = 1).dropna()
    print('Age and Sex:{}'.format(Age_Sex.shape[0]))
    print('Age and BMI:{}'.format(Age_BMI.shape[0]))
    print('Sex and BMI:{}'.format(Sex_BMI.shape[0]))

    Age_Sex_BMI = df.dropna()
    print('all metadeta:{}'.format(Age_Sex_BMI.shape[0]))
    dataset.append(Age)
    dataset.append(Sex)
    dataset.append(BMI)
    dataset.append(Age_Sex)
    dataset.append(Age_BMI)
    dataset.append(Sex_BMI)
    dataset.append(Age_Sex_BMI)
    
    dataset_id = pd.read_csv('../index.txt')
    for model_name in ['svm']:
        for data in dataset:

            info_meta = var_name(data)[0]

            for i in ['Age', 'BMI', 'Sex']:
                if i in data.columns:
                    data = data.drop(i, axis = 1)   

            dataset_res = pd.DataFrame(index=[], columns=['x','y','datasetname'] )
            disease_res = pd.DataFrame(index=[], columns=['x','y','datasetname'] )
            # leave one dataset out
            for i in range(dataset_id.shape[0]):
                # print(dataset_id['id'][i],var_name(data)[0])
                
                df_single_dataset = data.query('DiseaseState == "H" or DiseaseState == "{}"'.format(dataset_id['DiseaseState'][i]))
                df_single_dataset = df_single_dataset.query('LibraryName.str.contains("{}")'.format(dataset_id['id'][i]), engine='python')
                if df_single_dataset.shape[0] > 50:  
                    # df_single_dataset['DiseaseState']= LabelEncoder().fit_transform(df_single_dataset['DiseaseState'])
                    X = df_single_dataset.drop(["LibraryName", "DiseaseState"], axis = 1)
                    y = df_single_dataset[['DiseaseState']].values
                    y = np.where(y == 'H', 0, 1)
                    single_auc_score = cv_and_roc(model_name, X, y)
                
                if df_single_dataset.shape[0] > 50:
                    # leave one dataset out
                    df_leave_dataset = data[~data["LibraryName"].str.contains(dataset_id['id'][i])]
                    X_train = df_leave_dataset.drop(["LibraryName","DiseaseState"],axis = 1)
                    y_train = df_leave_dataset[['DiseaseState']].values 
                    y_train = np.where(y_train == 'H', 0, 1)
        
                    X_test = df_single_dataset.drop(["LibraryName","DiseaseState"],axis = 1)
                    y_test= df_single_dataset[['DiseaseState']].values
                    y_test = np.where(y_test == 'H',0,1)
                    score = leave_one_test_roc(X_train, y_train, X_test, y_test, model_name)
                    record = pd.Series([single_auc_score, score, dataset_id['id'][i]], index=disease_res.columns)
                    dataset_res = dataset_res.append(record, ignore_index=True)
                
                if df_single_dataset.shape[0] > 50:
                    # leave one disease out
                    df_leave_disease = data[data['DiseaseState'] != dataset_id['DiseaseState'][i]]
                    df_leave_disease = df_leave_disease[~((df_leave_disease['LibraryName'].str.contains(dataset_id['id'][i])) & (df_leave_disease['DiseaseState'] == 'H'))]

                    X_train = df_leave_disease.drop(["LibraryName", "DiseaseState"], axis = 1)
                    y_train = df_leave_disease[['DiseaseState']].values 
                    y_train = np.where(y_train == 'H', 0, 1)
        
                    X_test = df_single_dataset.drop(["LibraryName", "DiseaseState"], axis = 1)
                    y_test= df_single_dataset[['DiseaseState']].values
                    y_test = np.where(y_test == 'H',0,1)
                    score = leave_one_test_roc(X_train, y_train, X_test, y_test, model_name)
                    record = pd.Series([single_auc_score, score, dataset_id['id'][i]], index=disease_res.columns)
                    disease_res = disease_res.append(record, ignore_index=True)




            plot_compare_auc(dataset_res,'dataset',info_meta,'{}/none_of_meta'.format(model_name))
            dataset_res = pd.DataFrame(index=[], columns=['x','y','datasetname'] )

            plot_compare_auc(disease_res,'disease',info_meta,'{}/none_of_meta'.format(model_name))
            disease_res = pd.DataFrame(index=[], columns=['x','y','datasetname'] )