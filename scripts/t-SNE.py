import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style('darkgrid')
from load_data import load_data
from sklearn.cluster import DBSCAN

import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    df = load_data()
    X = df.drop(['Age', 'Sex', 'BMI', 'DiseaseState', 'LibraryName'], axis = 1)
    model = TSNE(n_components = 2, perplexity=50.0)
    tsne_result = model.fit_transform(X) 
    
    # `eps`は試行錯誤した結果
    dbscan = DBSCAN(eps=3)
    dbscan_tsne = dbscan.fit_predict(tsne_result)

    #いい感じに色を付ける
    color=cm.brg(np.linspace(0,1,np.max(dbscan_tsne) - np.min(dbscan_tsne)+1))
    for i in range(np.min(dbscan_tsne), np.max(dbscan_tsne)+1):
        plt.plot(tsne_result[dbscan_tsne == i][:,0],
                tsne_result[dbscan_tsne == i][:,1],
                ".",
                color=color[i+1]
                )
        plt.text(tsne_result[dbscan_tsne == i][:,0][0],
                tsne_result[dbscan_tsne == i][:,1][0],
                str(i), color="black", size=16
                )
    plt.savefig('../output/tsne.png')