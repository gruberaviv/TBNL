# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:33:14 2018

@author: Aviv Gruber
"""

from marg_prob import margprob
import numpy as np
import pandas as pd
from scipy.stats import entropy

def entropy_from_counts(counts):
    counts = np.asarray(counts)
    probs = counts/counts.sum()
    ent = -(probs[probs>0]*np.log2(probs[probs>0])).sum()
    return ent

def cond_entropy_from_counts(data, counts, target = None, MC = 0, xi=None):
    
    unique_data, unique_count = margprob(data, counts, data.columns, MC, xi)
    if target is None:
        return entropy_from_counts(unique_count)
    if MC > 0:
        if len(xi.shape) == 1:
            xi = pd.DataFrame([xi]).T
        response = np.repeat(0, MC)
        unique_data_prob = np.array(unique_count)/sum(unique_count)
        cumul_joint = np.cumsum(unique_data_prob)
        count_cumulated_margin = 0
        conditioning_set = set(data.columns) - set(flatten_list([target]))
        if len(conditioning_set) > 0:
            conditioning, cond_count = margprob(unique_data, unique_count, list(conditioning_set))
            prob_conditioning = np.array(cond_count)/sum(cond_count)
            cumul_conditioning = np.cumsum(prob_conditioning)
        for it in xi.index.tolist():      
            whichi = xi.loc[it,0] <= cumul_joint
            realization_joint_idx = whichi.tolist().index(True)
            if len(conditioning_set) > 0:
                whichita = xi.loc[it,0] <= cumul_conditioning
                realization_conditioning_idx = whichita.tolist().index(True)
                response[it] = np.log2(prob_conditioning[realization_conditioning_idx]) - np.log2(unique_data_prob[realization_joint_idx])
            else:
                response[it] = - np.log2(unique_data_prob[realization_joint_idx])           
        return np.mean(response)
    else:
        if len(set(data.columns) - set(flatten_list([target]))) == 0:
            return entropy_from_counts(unique_count)
        yumba, _ = margprob(unique_data, unique_count, list(set(data.columns) - set(flatten_list([target]))), 0)
        unique_data_yumba = yumba.drop_duplicates()
        yumba_set = set(data.columns) - set(flatten_list([target]))
        yumba_set = list(yumba_set)
        unique_data['counts'] = unique_count
        count_cumulated_margin = unique_data.groupby(yumba_set)["counts"].sum().to_list()
        ent = unique_data.groupby(yumba_set)['counts'].apply(lambda x: entropy(x, base=2)).to_list()
        prob = np.asarray(count_cumulated_margin) / sum(np.asanyarray(count_cumulated_margin))
    return sum(x * y for x, y in zip(ent, prob))

def mutual_information_from_counts(data, counts, target, about = None, MC = 0, xi = None):
  unique_data, unique_count = margprob(data, counts, data.columns, MC)
  data_4_entropy, _ = margprob(unique_data, unique_count, target, MC)
  if about is None:
      print("Mutual information must have two variables (about cannot be null")
      return None
  conditioned_colnames = list(set(list(data)) - set(flatten_list([about] + [target])))
  data_4_condentropy, counts_4_condentropy = margprob(unique_data, unique_count, flatten_list([conditioned_colnames] + [target]), MC)
  if MC > 0:
      if xi is None:
          xi = np.random.rand(MC)
  H_Y_conditioned_X = cond_entropy_from_counts(data_4_condentropy, counts_4_condentropy, target, MC, xi)
  H_Y_conditioned_Z_X = cond_entropy_from_counts(data=unique_data, counts=unique_count, target=target, MC=MC, xi=xi)
  MI = H_Y_conditioned_X - H_Y_conditioned_Z_X
  return(MI)

def flatten_list(the_list):
    flat_list = []
    for item in the_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list
