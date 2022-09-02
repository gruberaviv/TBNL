# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:57:23 2018

@author: Aviv Gruber
"""

from entropy_from_counts import *
import numpy as np
import pandas as pd
import networkx as nx

from pyvis.network import Network
import matplotlib.pyplot as plt
  
def feature_selection_from_counts(data, counts, target, local_constraints, graph = None, MC = 0):  
  features = list()
  total_info = 0
  if local_constraints is not None:
      dic = local_constraints
  keep_going = False
  if len(flatten_list(list(set(data.columns.tolist()) - set(flatten_list([target]))))) > 0:
      keep_going = True
  if target not in data.columns:
      print(["Feature selection Hantarish returned Nothing since the target was not found in the data. data fields: ", data.columns.tolist()])
      return features, graph
  if MC > 0:
      xi = np.random.rand(MC)
  else:
      xi = None
  target_marg, target_count = margprob(data, counts, target, MC, xi)
  target_entropy = cond_entropy_from_counts(target_marg, target_count, MC=MC, xi=xi)
  print("Node =", target, "Entropy =", target_entropy, sep=" ")
  if target_entropy == 0:
      return features, graph
  descendants = list(nx.descendants(graph, target))
  directed = True
  g4 = Network(directed=directed, height="600px", width="1000px", heading='TBNL')#, layout='hierarchical')
  g4.set_edge_smooth('dynammic')
  g4.from_nx(graph)
  g4.show_buttons(filter_=['physics'])
  g4.barnes_hut(gravity=-18550, central_gravity=0.75, spring_length=205)
  g4.show('TBNL.html')
  i = -1
  while keep_going:
      relevant_cols = list(set(data.columns) - set(flatten_list([descendants] + [features] + [target])))
      if len(relevant_cols) == 0:
          keep_going = False
          break
      MI = list()
      for var in relevant_cols:
          MI.append((var, mutual_information_from_counts(data[flatten_list([features] + [var] + [target])], counts, target, var, MC, xi)))
      feats, infos = zip(*MI)
      info_gain_max = max(infos)
      feature_max = feats[infos.index(info_gain_max)]
      if 'min_info_gain' in dic.keys():
          if info_gain_max/target_entropy < dic['min_info_gain']:
              keep_going = False
              break  
      total_info_temp = total_info + info_gain_max
      if 'max_total_info' in dic.keys():
          if total_info_temp/target_entropy > dic['max_total_info']:
              keep_going = False
              break 
      if info_gain_max > 0 and total_info/target_entropy <= 1:
          features.append([feature_max][0])
          total_info = total_info + info_gain_max     
          if graph is not None:
              graph.add_edge(feature_max, target)
              g4.from_nx(graph)
              g4.show('TBNL.html')
          print(features[-1], round(info_gain_max/target_entropy*100, 2),"% (",round(total_info/target_entropy*100, 2),"%)")
          if 'max_features' in dic.keys():
              if len(list(graph.predecessors(target))) >= dic['max_features']:
                  keep_going = False
          if 'max_total_info' in dic.keys():
              if total_info/target_entropy >= dic['max_total_info']:
                  keep_going = False
          i = i + 1
          if i == len(list(set(data.columns.tolist()) - set(flatten_list([target])))):
              keep_going = False
      else:
          keep_going = False
  return features, graph
 
