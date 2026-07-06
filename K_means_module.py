import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def run_kmeans(df, clusterNum=5):
    df=df.drop('CustomerID', axis=1)
    
    le=LabelEncoder()
    df['Gender']=le.fit_transform(df['Gender'])
    
    plt.figure(figsize=(10,6))
    plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],alpha=0.6)
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.grid(True,alpha=0.3)
    plt.show(block=False)
    
    x = df.values[:,0:]
    x = np.nan_to_num(x)
    
    kmeans = KMeans(init='k-means++',n_clusters=clusterNum,n_init=12)
    kmeans.fit(x)
    labels=kmeans.labels_
    
    df["cluster_km"]=labels
    print(df.groupby('cluster_km').mean())
    
    plt.figure(figsize=(10,6))
    plt.scatter(x[:,2], x[:,3],s=50, c=labels.astype(np.float64), alpha=0.5)
    plt.xlabel('Annual Income', fontsize=15)
    plt.ylabel('Spending Score',fontsize=15)
    plt.show()
    
    return df, labels