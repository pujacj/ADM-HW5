#!/usr/bin/env python
# coding: utf-8

# ### Functionality 1 - Find the Neighbours!
# It takes in input:
# 
# a node v
# 
# One of the following distances function: t(x,y), d(x,y) or network distance (i.e. consider all edges to have weight equal to 1).
# 
# a distance threshold d
# 
# Implement an algorithm (using proper data structures) that returns the set of nodes at distance <= d from v, corresponding to v’s neighborhood.

# In[1]:


import pandas as pd
#import the datasets with the informations about the distances between the nodes and their coordinates
node_info= pd.read_csv("node_information_file.csv")
travel_time= pd.read_csv("travel_time_graph.csv")
spatial_distance= pd.read_csv("distance_graph.csv")
node_info.columns=["id_node", "lat", "long"]

spatial_distance.columns=['Node1', 'Node2', 'distance']
travel_time.columns=['Node1', 'Node2', 'distance']

#we create the dataset with the network distance(1)
from pandas import DataFrame
network_df=DataFrame(columns= ['Node1', 'Node2','network_distance'])
network_df['Node1'] = pd.Series(travel_time['Node1'])
network_df['Node2'] = pd.Series(travel_time['Node2'])
network_df['network_distance']=1
network_df.columns=['Node1', 'Node2', 'distance']


# In[2]:


#we divide the latitude and longitude values by one million so that they can be used
node_info['lat'] = node_info['lat']/(1000000)
node_info['long'] = node_info['long']/(1000000)


# In[3]:


#the algorithm that gives us the set of nodes that are neighbors of a given starting node, according to a specific limit

v= int(input("Choose a starting node: "))
distance=input("Choose a type of distance: ")
max_depth=int(input("Insert a treshold value: "))   
    
def search_neighbour(v, distance, max_depth): 
    
    if distance == "spatial":   #creating a local df 
        df = spatial_distance
    elif distance == "time":
        df = travel_time
    elif distance == "network":
        df = network_df

    neighbour = df[(df['Node1'] == v) & (df['distance'] <= max_depth)]
    l_n = len(neighbour)

    if l_n>0:
        listNodes2 = neighbour['Node2'].tolist()
        for nodo2 in listNodes2:
            d = df[(df['Node1'] == v) & (df['Node2'] == nodo2)]
            d=int(d['distance'])
            neighbour=pd.concat([neighbour,search_neighbour(nodo2, distance, max_depth-d)]).drop_duplicates().reset_index(drop=True)
    return neighbour



neighbour=search_neighbour(v,distance,max_depth)
result = set(neighbour['Node2'].tolist())

print(result)


# ### Visualization 1 - Find the Neighbours!
# Once the user runs Functionality 1, we want the system to show in output a complete map that contains: the input node, the output nodes and all the streets that connect these points. Choose different colors in order to highlight which is the input node, and which are the output nodes. Furthermore, choose different colors for edges, according to the distance function used.

# In[4]:


import json
import urllib
import pandas as pd 
import numpy as np
from io import StringIO 
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import collections
import heapq
import geopy
from geopy.geocoders import Nominatim, GoogleV3
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import json, string
import requests
import geocoder
from folium import Map, Marker, GeoJson, LayerControl
from ediblepickle import checkpoint
from tqdm import tqdm_notebook
import os 
import folium
from folium import plugins
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


node_info['Node1']=node_info['id_node']
neighbour=pd.merge(node_info, neighbour, on='Node1', how='inner')


# In[6]:


node_info['Node2']=node_info['id_node']


# In[7]:


neighbour.rename(columns={'Node2_y':'Node2'}, inplace=True)


# In[8]:


neighbour=pd.merge(node_info,neighbour, on='Node2', how='inner')


# In[9]:


# neighbour


# In[10]:


map = folium.Map(location=[np.median((neighbour['long_x']).tolist()),
                           np.median((neighbour['lat_x']).tolist())], 
                 default_zoom_start=120)#keep the median to localize our data

for i in tqdm_notebook(range(len(neighbour))):
    
    folium.features.RegularPolygonMarker(location = [(neighbour['long_x'].values)[0], 
                                                     (neighbour['lat_x'].values)[0]],
                                             number_of_sides = 3,
                                             popup='Starting node',
                                             fill_color='red',
                                             radius = 15,
                                             weight = 4,
                                             fill_opacity = 0.8).add_to(map)

    

    


    folium.features.RegularPolygonMarker(location = [(neighbour['long_x'].values)[i], 
                                                         (neighbour['lat_x'].values)[i]],
                                                 number_of_sides = 3,
                                                 radius = 10,
                                                 weight = 4,
                                                 fill_opacity = 0.8).add_to(map)

#to add the edge between two nodes
for i in tqdm_notebook(range(len(neighbour))):
    folium.PolyLine(locations = [(((neighbour['long_x']).values)[i], 
                                  ((neighbour['lat_x']).values)[i]), 
                                 (((neighbour['long_y']).values)[i], 
                                  ((neighbour['lat_y']).values)[i])], 
                                color="red",
                                line_opacity = 0.5).add_to(map) 

map.save("mapneighbours.html") #saving the map in html
 
map


# In[ ]:




