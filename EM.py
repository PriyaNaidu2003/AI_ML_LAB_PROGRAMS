import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
data=pd.read_csv('em.csv')
print("Input data end shape")
print(data.shape)
data.head()
f1=data['v1'].values
f2=data['v2'].values
x=np.array(list(zip(f1,f2)))
print("X",x)
print("Graph for whole data set")
plt.scatter(f1, f2, c="blue",s=7)
plt.show()
kmean=KMeans(20,random_state=0)
labels=kmean.fit(x).predict(x)
print("labels:",labels)
Centroids=kmean.cluster_centers_
print("centroids:",Centroids)
plt.scatter(x[:,0], x[:,1], c=labels,s=40,cmap='viridis')
print("Graph using kmeans algo")
plt.scatter(Centroids[:,0],Centroids[:,1], marker='*',s=200,c='#050505')
plt.show()
gm=GaussianMixture(n_components=3).fit(x)
labels=gm.predict(x)
prob=gm.predict_proba(x)
size=10*prob.max(1)**3
print('Graph using em')
plt.scatter(x[:,0], x[:,1], c=labels,s=size,cmap='viridis')
plt.show()
   
