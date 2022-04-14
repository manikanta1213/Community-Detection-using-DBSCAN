#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
from igraph import*
import igraph
import numpy as np
import pandas as pd
import sklearn


# # Importing datatset - FACEBOOK STANFORD NETWORK DATASET
# Initial network of facebook data

# In[2]:


f=Graph.Read_Edgelist(r"C:\Users\manik\Desktop\facebook_combined.txt",directed=None)
visual_style = {}
visual_style["vertex_size"] = 5
visual_style["margin"] = 17
visual_style["bbox"] = (250,250)
my_layout =f.layout_fruchterman_reingold()
visual_style["layout"] = my_layout
#igraph.plot(f,r"C:\Users\manik\Desktop\graph plots\initial_network.png",**visual_style)
print(f)


# #converting text file to csv and no of edges in the network before applying dbscan

# In[96]:


a=[]
b=[]
for i in f.get_edgelist():
    a.append(i[0])
    b.append(i[1])


# In[97]:


x=pd.DataFrame()


# In[98]:


x['A']=a
x['B']=b


# In[99]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


# Using KNN to find out the optimal epsillon value.
# The optimal value for epsilon will be found at the point of maximum curvature.

# In[100]:


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(x)
distances, indices = nbrs.kneighbors(x)


# In[101]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(3,3))
plt.plot(distances)


# # Applying dbscan on facebook data

# In[102]:


from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=6,min_samples=6)


# In[103]:


model=dbscan.fit(x)
model


# In[71]:


labels=model.labels_


# # Total clusters after performing dbscan

# In[72]:


n_clusters=len(set(labels))- (1 if -1 in labels else 0)
n_clusters


# In[73]:


#print(length of noise points)
c=0
ans=[]
for i in labels:
    if i==-1:
        c+=1
    ans.append(i)
print("number of noise points",c)


# In[42]:


#import networkx as nx
#g= nx.read_edgelist(r"C:\Users\manik\Desktop\facebook_combined.txt")
#plt.figure(figsize=(10,9))
#nx.draw(g)


# In[ ]:





# In[43]:


#new_graph=Graph()


# # Removing all the outliers i.e; edges in the network

# In[74]:


j=0
for node in f.get_edgelist():
    if labels[j]==-1:
        f.delete_edges(f.get_eid(node[0],node[1]))
    j+=1


# In[91]:


new_labels = np.delete(labels, np.where(labels == -1))


# In[92]:


len(f.get_edgelist())


# In[93]:


a=[]
b=[]
for i in f.get_edgelist():
    a.append(i[0])
    b.append(i[1])
print(len(f.get_edgelist()))
Y=pd.DataFrame()


# In[48]:


Y['A']=a
Y['B']=b


# In[139]:


#silhouette score after performing dbscan
from sklearn import metrics
print("silhoutte score:",metrics.silhouette_score(x, labels))


# In[140]:


#silhouette score after performing dbscan
from sklearn import metrics
print("silhoutte score:",metrics.silhouette_score(Y, new_labels))


# # Network figure after removing outlier edges

# In[22]:


#igraph.plot(f,r"C:\Users\manik\Desktop\graph plots\final1.png")


# In[49]:


print(len(f.get_edgelist()))


# In[50]:


#igraph.plot(f)


# In[25]:


sample_cores=np.zeros_like(labels,dtype=bool)
sample_cores[dbscan.core_sample_indices_]=True
sample_cores


# In[26]:


plt.figure(figsize=(10,6))
plt.scatter(x['A'], x['B'],c=labels, cmap='Paired')
plt.title("Clusters determined by DBSCAN")


# # Applying fast greedy to detect communities

# In[77]:


clusters=f.community_fastgreedy()
print(clusters)


# In[78]:


clusters=clusters.as_clustering()


# In[90]:


print(clusters)


# In[89]:


clusters.modularity


# In[88]:


clusters.membership


# In[83]:


pal=igraph.drawing.colors.ClusterColoringPalette(len(clusters))


# In[84]:


f.vs['color']=pal.get_many(clusters.membership)


# # detected clusters

# In[85]:


visual_style = {}
visual_style["vertex_size"] = 5
visual_style["margin"] = 17
visual_style["bbox"] = (250,250)
my_layout =f.layout_fruchterman_reingold()
visual_style["layout"] = my_layout


# In[87]:


graph.plot(f,r"C:\Users\manik\Desktop\graph plots\edge removal minsamples=18.png",**visual_style)


# In[42]:



def assortativity(graph, degrees=None):
    if degrees is None: degrees = graph.degree()
    degrees_sq = [deg**2 for deg in degrees]
 
    m = float(graph.ecount())
    num1, num2, den1 = 0, 0, 0
    for source, target in graph.get_edgelist():
        num1 += degrees[source] * degrees[target]
        num2 += degrees[source] + degrees[target]
        den1 += degrees_sq[source] + degrees_sq[target]
 
    num1 /= m
    den1 /= 2*m
    num2 = (num2 / (2*m)) ** 2
 
    return (num1 - num2) / (den1 - num2)

print("Assortativity of the graph:", assortativity(f))


# In[43]:


cliques = f.cliques(min=3, max=3)
triangle_count = [0] * f.vcount()
for i, j, k in cliques:
    triangle_count[i] += 1
    triangle_count[j] += 1
    triangle_count[k] += 1

print("Average number of triangles:", sum(triangle_count)/f.vcount())
print("Maximum number of triangles:", max(triangle_count))
print("Vertex ID with the maximum number of triangles:", triangle_count.index(max(triangle_count)))


# In[44]:


print("Density of the graph:", 2*f.ecount()/(f.vcount()*(f.vcount()-1)))


# In[46]:


degrees = []
total = 0
n_vertices = 4039
for n in range(n_vertices):
    neighbours = f.neighbors(n, mode='ALL')
    total += len(neighbours)
    degrees.append(len(neighbours))
    
print("Average degree:", total/n_vertices)
print("Maximum degree:", max(degrees))
print("Vertex ID with the maximum degree:", degrees.index(max(degrees)))


# In[ ]:




