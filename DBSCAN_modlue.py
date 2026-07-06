import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.cm as cm

def find_best_epsilon(x_scaled, n_neighbors=5):

    neigh=NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(x_scaled)
    distan,inde=neigh.kneighbors(x_scaled)
    k_disran=distan[:,-1]
    k_disran=np.sort(k_disran)

    kneedle=KneeLocator(range(len(k_disran)), k_disran, curve='convex',direction='increasing')
    return kneedle.knee_y

def run_dbscan():
    df=pd.read_csv(r"T:\VS_project\ML_Jadi\3rd_project\1632560262896716.csv")
    #print(df.head())
    df = df.drop('CustomerID', axis=1)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    
    x = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    x_scaled=StandardScaler().fit_transform(x)   
    #print(x_scaled)
    #print(find_best_epsilon(x_scaled,n_neighbors=5))
    epsilon=find_best_epsilon(x_scaled,n_neighbors=5)*0.38
    #epsilon=0.8
    minimumSamples=5
    db=DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(x_scaled)
    labels=db.labels_    
    df['cluster']=labels
    #print(df)

    uniq_labels=set(labels)
    colors=cm.rainbow(np.linspace(0,1,len(uniq_labels)))

    plt.figure(figsize=(15,11))

    for k, col in zip(uniq_labels, colors):
        if k == -1:
            current_color = [[0, 0, 0, 1]] 
            current_label = 'Noise'
            current_marker = 'x'
            s_size = 50
        else:
            current_color = [col]       
            current_label = f'Cluster {k}'
            current_marker = 'o'
            s_size = 50

        member_mask = (df['cluster'] == k)
        xy = df[member_mask]

        plt.scatter(xy['Annual Income (k$)'], xy['Spending Score (1-100)'],
                    s=s_size, c=current_color,
                    marker=current_marker, label=current_label)

    plt.legend()
    plt.title('DBSCAN Result')
    plt.show()

#run_dbscan()