import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pylab
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm

def run_hierarchical(df_input):
    df=df_input.copy()
    
    df=df.drop('CustomerID', axis=1)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    freature_set =df[['Annual Income (k$)', 'Spending Score (1-100)']]

    x = freature_set.values
    min_max_scalar = MinMaxScaler()
    freature_mtx = min_max_scalar.fit_transform(x)

    D = euclidean_distances(freature_mtx)
    z = linkage(freature_mtx, method='complete', metric='euclidean')
    max_d = 3
    clusters = fcluster(z, max_d, criterion='distance')

    fig =pylab.figure(figsize=(18, 50))

    def llf(id):
        return '[%s %s %s]' % (df['Gender'].iloc[id], df['Age'].iloc[id], int(float(df['Spending Score (1-100)'].iloc[id])))

    dendro = hierarchy.dendrogram(z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12)
    # pylab.show() 

    agglom = AgglomerativeClustering(n_clusters=5, linkage='complete')
    agglom.fit(freature_mtx)

    df['cluster_'] = agglom.labels_

    n_clusters = max(agglom.labels_) + 1
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_labels = list(range(0, n_clusters))

    plt.figure(figsize=(16, 14))

    for color, label in zip(colors, cluster_labels):
        subset = df[df.cluster_ == label]
        plt.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'], 
                    s=50, c=[color], label='cluster ' + str(label))

    plt.legend()
    plt.title('Clusters (Hierarchical)')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

