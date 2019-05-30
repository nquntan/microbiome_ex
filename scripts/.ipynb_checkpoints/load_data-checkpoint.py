import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tools import *

def load_data():
    count = pd.read_csv("../microbiomeHD_combined.count.genus.level.csv",index_col = 0)
    meta = pd.read_csv("../metadata_combined.csv")
    count = count.T
    #####相対カウントを用いるとき#########################
    count = count.apply(lambda x:x/sum(x) if sum(x) != 0 else x,axis=1)
    ###############################################

    #データの前処理(先行研究のとこ参照)
    d_m = pd.merge(meta, count, left_on='LibraryName', right_index=True).dropna(axis = 0, how = 'all')
    selected = d_m['LibraryName'].str.contains('ibd_huttenhower')  
    d_m.loc[selected, 'DiseaseState'] = d_m.loc[selected, 'DiseaseState'].replace('UC', 'CD')
    selected = d_m['LibraryName'].str.contains('ibd_engstrand_maxee')  
    d_m.loc[selected, 'DiseaseState'] = d_m.loc[selected, 'DiseaseState'].replace('UC', 'CD')
    selected = d_m['LibraryName'].str.contains('ibd_alm')  
    d_m.loc[selected, 'DiseaseState'] = d_m.loc[selected, 'DiseaseState'].replace('UC', 'CD')
    selected = d_m['LibraryName'].str.contains('ibd_alm')  
    d_m.loc[selected, 'DiseaseState'] = d_m.loc[selected, 'DiseaseState'].replace('nonIBD', 'H')
    selected = d_m['LibraryName'].str.contains('ibd_gevers_2014')  
    d_m.loc[selected, 'DiseaseState'] = d_m.loc[selected, 'DiseaseState'].replace('nonIBD', 'H')
    selected = d_m['LibraryName'].str.contains('ra_littman')  
    d_m.loc[selected, 'DiseaseState'] = d_m.loc[selected, 'DiseaseState'].replace('PSA', 'RA')
    d_m = d_m.dropna(subset=['DiseaseState'])
    
    d_m['Sex'] = d_m['Sex'].astype('category')
    d_m['DiseaseState'] = d_m['DiseaseState'].astype('category')
    d_m['DiseaseState']= LabelEncoder().fit_transform(d_m['DiseaseState'])
    return d_m