#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt
import random
from igraph import*
import igraph


# In[35]:


import pandas as pd
x=pd.read_csv(r'C:\Users\manik\Desktop\facebook.csv')


# In[36]:


x.head()


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[38]:


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=20)
nbrs = neigh.fit(x)
distances, indices = nbrs.kneighbors(x)


# In[39]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(3,3))
plt.plot(distances)


# In[74]:


from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=10,min_samples=16)


# In[75]:


model=dbscan.fit(x)
model


# In[51]:


labels=model.labels_
ans=[]
for i in labels:
    ans.append(i+1) 


# In[52]:


len(labels)


# In[53]:


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# In[12]:


#silhouette score after performing dbscan
from sklearn import metrics
h=metrics.silhouette_score(x, labels)
print(h)


# In[396]:


#res.append((n_clusters_,n_noise_,h))


# In[397]:


f=Graph.Read_Edgelist(r"C:\Users\manik\Desktop\facebook_combined.txt",directed=None)
pal=igraph.drawing.colors.ClusterColoringPalette(len(set(ans)))
f.vs['color']=pal.get_many(ans)


# In[398]:


visual_style = {}
visual_style["vertex_size"] = 5
visual_style["margin"] = 17
visual_style["bbox"] = (250,250)
my_layout =f.layout_fruchterman_reingold()
visual_style["layout"] = my_layout
igraph.plot(f,r"C:\Users\manik\Desktop\graph plots\dbscan+mins=20.png",**visual_style)


# In[92]:


if color_name_to_rgba("red")==f.vs[0]["color"]:
    print("ALL red points are noise points")


# In[399]:


k=pd.DataFrame(res)
k.to_csv(r"C:\Users\manik\Desktop\dbscan.csv")


# # Removing Nodes

# In[54]:


import networkx as nx
g=nx.read_edgelist(r"C:\Users\manik\Desktop\facebook_combined.txt")
g1=nx.read_edgelist(r"C:\Users\manik\Desktop\facebook_combined.txt")
c=0
k=[]
for i in g1.nodes():
    if labels[c]==-1:
        g.remove_node(i)
        k.append(i)
    c+=1
a=[]
b=[]
for i in g.edges():
    a.append(i[0])
    b.append(i[1])
y=0
for i in g1.edges():
    if i[0] in k or i[1] in k:
        y+=1


# In[55]:


print("edges removed",y)


# In[56]:


import scipy as sp
A = nx.adjacency_matrix(g)
res = np.delete(labels, np.where(labels == -1))


# In[57]:


#modularity in dbscan
from sknetwork.clustering import modularity
from sknetwork.data import house
import numpy as np
np.round(modularity(A,res),2)


# In[58]:


import pandas as pd
data=pd.DataFrame()


# In[59]:


data['A']=a
data['B']=b
data.head()


# In[60]:


ans1=[]
for i in labels:
    if i!=-1:
        ans1.append(i)


# In[61]:


data.to_csv(r"C:\Users\manik\Desktop\facebook_deleted.txt", header=None, index=None, sep=' ', mode='a')


# In[62]:


u=Graph.Read_Edgelist(r"C:\Users\manik\Desktop\facebook_deleted.txt",directed=None)


# In[417]:


visual_style = {}
visual_style["vertex_size"] = 5
visual_style["margin"] = 17
visual_style["bbox"] = (250,250)
my_layout =u.layout_drl()
visual_style["layout"] = my_layout
igraph.plot(u,r"C:\Users\manik\Desktop\graph plots\dbscan_node_removal_min_Samples=.png",**visual_style)


# In[63]:


clusters=u.community_fastgreedy()
print(clusters)


# In[64]:


clusters=clusters.as_clustering()
print(clusters)


# In[65]:


print(clusters.modularity)


# In[66]:


print(clusters.membership)


# In[67]:


pal=igraph.drawing.colors.ClusterColoringPalette(len(clusters))


# In[68]:


u.vs['color']=pal.get_many(clusters.membership)


# In[69]:


visual_style = {}
visual_style["vertex_size"] = 5
visual_style["margin"] = 17
visual_style["bbox"] = (250,250)
my_layout =u.layout_drl()
visual_style["layout"] = my_layout
igraph.plot(u,r"C:\Users\manik\Desktop\graph plots\dbscan_node_removal_output_minsamples=16.png",**visual_style)


# In[ ]:




