import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
points=np.array([[3,7],[4,6],[5,5],[6,4],[7,3],[6,2],[7,2],[8,4],[3,3],[2,6],[3,5],[2,4]])
plt.figure(figsize=(6,6))
plt.scatter(points[:,0],points[:,1],color='b')
plt.title("Row data points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
db=DBSCAN(eps=1.9,min_samples=4).fit(points)
labels=db.labels_
core_points_mask=np.zeros_like(labels,dtype=bool)
core_points_mask[db.core_sample_indices_]=True
border_points_mask=(labels!=-1) & ~core_points_mask
noise_points_mask=(labels==-1)
plt.figure(figsize=(6,6))
plt.scatter(points[core_points_mask,0],points[core_points_mask,1],color='red',marker='o',label='core points')
plt.scatter(points[border_points_mask,0],points[border_points_mask,1],color='green',marker='o',label='border points')
plt.scatter(points[noise_points_mask,0],points[noise_points_mask,1],color='black',marker='x',label='noise points')
for p in points[core_points_mask]:
    c=plt.Circle((p[0],p[1]), 1.9,color='blue',fill=False,linestyle='dotted')
    plt.gca().add_artist(c)
plt.title("DBSCAN clustering with approxbondraies")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis('equal')
plt.show() 
print("DBSCAN label:",labels)
