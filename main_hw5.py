#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
import networkx as nx


# In[3]:


data_dist = pd.read_csv(r'C:\Users\danyl\Downloads\Nouveau dossier\USA-road-d.CAL.gr',sep = " ",header = None)
data_time = pd.read_csv(r'C:\Users\danyl\Downloads\Nouveau dossier\USA-road-t.CAL.gr',sep = " ", header = None)
df_data = pd.read_csv(r'C:\Users\danyl\Downloads\Nouveau dossier\USA-road-d.CAL.co',sep = " ", header = None)
nodes = []
lat= []
long = []
for key, value in df_data.iterrows():
    nodes.append(value[1])
    lat.append(value[2]/1000000)
    long.append(value[3]/1000000)
import networkx as nx
G=nx.Graph()
for key,value in data_dist.iterrows():
    G.add_edge(value[1], value[2], dist=value[3] )
def get_shortest_path(graph, start,end):
    G = graph
    to_visit = [start] 
    path = []
    i = 0
    visited = [] #save the nodes that has been visited 
    while end not in to_visit: #the stopping criteria is when the final node is in the visit list
        if to_visit[i] not in visited: #we visit each node just one time

            for k in list(G.neighbors(to_visit[i])): 
                to_visit.append(k) #we add the neighbors of the node we are visiting in the 'to_visit' list
            path.append(list(G.neighbors(to_visit[i]))) #we save the path to know where each node is from
            visited.append(to_visit[i]) #we add the node we just visited to the 'visited' list
        i = i + 1
    element = visited[-1] # We save the element that has added the 'start' point in the 'to_visit' list
    i = 1
    result = [end, element]
    while i <100000 and element !=start: #because we are getting the path in the opposite direction, we iterate until finding the start node
        p = []
        for k in range(len(path)):
            for i in range(len(path[k])):
                if path[k][i] == element: #we look for the last element we visited
                    p.append(k)
        element = visited[p[0]] # here we get which element is its parent
        i = i+1
        result.append(element) # we add each parent to the result because its part of the path
    result.reverse()
    return(result)
def get_distance_nodes(function, start, end):
    a = function.loc[(function[1] == start) & (function[2] == end)] #use the dataframe to get the distance
    if len(a)>0 : 
        return int(a[3]) #get the distance
    else: 
        return 0
def get_distance(function, start, end):
    path = get_shortest_path(G, start,end) #using the get_shortest function
    i =0
    sum = 0
    while i < len(path)-1:
        sum = sum + get_distance_nodes(function, path[i], path[i+1]) #getting the distance between all connected node in the path
        i = i+1
        
    return(sum)
from itertools import permutations
def f2(list_nodes, function):
    l = []
    for s in permutations(list_nodes): #we try all the possibilities of the initial list of nodes
        case = list(s)
        sum_tot = 0
        i = 0
        while i < len(case)-1:
            sum_tot = sum_tot + get_distance(function, case[i], case[i+1]) #we calculate the distance for each possibility
            i = i +1
        element = (case, sum_tot)
        l.append(element) # we save the for each possibility, the distance
    l = sorted(l, key=lambda tup: (tup[1]),reverse = False) # we sort the list
    best_net = l[0][0] # we get the best possibilities
    
    

    i = 0
    list_edge = []
    while i < len(best_net)-1:
            list_edge = list_edge + get_shortest_path(G, best_net[i],best_net[i+1])
            i = i+1
    return(best_net, list_edge) # we return the best path


    
def visu_2(G,list_nodes,function):
    net, result = f2(list_nodes, function) # use the previous function
    edges = []
    for k in result:
        for i in list(G.neighbors(k)):
            edges.append((k,i)) #we fill the list with the neighbors of the nodes in the result
    r = []
    i = 0
    while i < len(result) -1:
        edges.append((result[i],result[i+1])) #we full the list with the nodes in the result
        r.append((result[i],result[i+1])) # we fill another list to print it with a different color
        i = i+1
    fig = plt.gcf()
    fig.set_size_inches(30, 30)


    G1 = nx.DiGraph()
    G1.add_edges_from(edges)

    val_map = {1: 2.0,
               }

    values = [val_map.get(node, 0.2) for node in G1.nodes()]

    # Specify the edges you want here

    red_edges = r
    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in G1.edges()]
    black_edges = [edge for edge in G1.edges() if edge not in red_edges]


    pos = nx.spring_layout(G1)
    
    for key,value in pos.items() :
        index = nodes.index(key)
        value[0] = lat[index]
        value[1] = long[index] # we get the real coordinates of each node

    
    nx.draw_networkx_nodes(G1, pos, cmap=plt.get_cmap('jet'), 
                           node_color = values, node_size = 1)
    nx.draw_networkx_labels(G1, pos)
    nx.draw_networkx_edges(G1, pos, edgelist=red_edges, edge_color='red', arrows=True)
    nx.draw_networkx_edges(G1, pos, edgelist=black_edges, edge_color='black',arrows=False)
    plt.show()
    
        
        
    
        




# In[ ]:





# In[6]:


def f4(G, H, list_nodes, function):
    l = []
    start = [H] # H is the starting point
    end = [list_nodes[-1]]
    list_nodes = list_nodes[:-1] 
    for s in permutations(list_nodes): #we try all the possibilities of the initial list of nodes
        case = start + list(s) + end
        sum_tot = 0
        i = 0
        while i < len(case)-1:
            sum_tot = sum_tot + get_distance(function, case[i], case[i+1]) #we calculate the distance for each possibility
            i = i +1
        element = (case, sum_tot)
        l.append(element) # we save the for each possibility, the distance
    l = sorted(l, key=lambda tup: (tup[1]),reverse = False) # we sort the list
    best_net = l[0][0] # we get the best possibilities

    i = 0
    list_edge = []
    while i < len(best_net)-1:
            list_edge = list_edge + get_shortest_path(G, best_net[i],best_net[i+1])
            i = i+1
    return(list_edge) # we return the best path


    
def visu_4(G,H, list_nodes,function):
    result = f4(G, H,list_nodes, function) # use the previous function
    edges = []
    for k in result:
        for i in list(G.neighbors(k)):
            edges.append((k,i)) #we fill the list with the neighbors of the nodes in the result
    r = []
    i = 0
    while i < len(result) -1:
        edges.append((result[i],result[i+1])) #we full the list with the nodes in the result
        r.append((result[i],result[i+1])) # we fill another list to print it with a different color
        i = i+1  
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)


    G1 = nx.DiGraph()
    G1.add_edges_from(edges)

    val_map = {1: 2.0,
               }

    values = [val_map.get(node, 0.25) for node in G1.nodes()]

    # Specify the edges you want here

    red_edges = r
    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in G1.edges()]
    black_edges = [edge for edge in G1.edges() if edge not in red_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G1)
    
    for key,value in pos.items() :
        index = nodes.index(key)
        value[0] = lat[index]
        value[1] = long[index] # we get the real coordinates of each node
    nx.draw_networkx_nodes(G1, pos, cmap=plt.get_cmap('jet'), 
                           node_color = values, node_size = 1)
    nx.draw_networkx_labels(G1, pos)
    nx.draw_networkx_edges(G1, pos, edgelist=red_edges, edge_color='red', arrows=True)
    nx.draw_networkx_edges(G1, pos, edgelist=black_edges, edge_color='black',arrows=False)
    plt.show()
    


# In[9]:


a = int(input("Choose the functionality (2 or 4)"))
lst = []
if a == 2:
    n = int(input("Enter number of nodes you want to visit "))
    for i in range(0, n): 
        ele = int(input('node you want to visit')) 
  
        lst.append(ele)
    visu_2(G,lst,data_dist)
else : 
    n = int(input("Enter number of nodes you want to visit "))
    b = int(input("Choose your starting node "))
    for i in range(0, n): 
        ele = int(input('node you want to visit')) 
  
        lst.append(ele)
    visu_4(G,b, lst,data_dist)
    


# In[ ]:




