import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import K_means_module
import Hierarchical_module
import DBSCAN_modlue
df=pd.read_csv(r"T:\VS_project\ML_Jadi\3rd_project\1632560262896716.csv")

while True:
    a=int(input('Which algorithm do you want to run?(1/2/3)' \
    '' \
    '\n       1.KMeans' \
    '\n       2.Hierarchical' \
    '\n       3.DBSCAN' \
    '\n       your answere= '))
    print(a)
    if a==1:
        K_means_module.run_kmeans(df,5)
        
    elif a==2:
        Hierarchical_module.run_hierarchical(df)
        
    elif a==3:
        DBSCAN_modlue.run_dbscan()

    else:
        print('invalid choice')

    c=input(' do you want to continue? (y/n)')
    print(c)
    if c=='n':break

    