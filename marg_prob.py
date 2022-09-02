# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:45:36 2018

@author: Aviv Gruber
"""

from os import name
import numpy as np
import pandas as pd

def margprob_old(data, counts = None, margin = None): 
    if type(data) == pd.Series:
        data = pd.DataFrame(data)
    if counts is None:
        counts = np.repeat(1, data.shape[0])
    if margin is None:
        data_margin = data
    else:
        if type(margin) == str:
            margin = [margin]
        data_margin = data[margin]
    cols = data_margin.columns
    temp_df = data_margin.copy()
    temp_df['counts'] = counts
    gb = temp_df.groupby(list(cols), observed=True)
    gbs = gb.counts.sum().reset_index(name='sum')
    unique_data_margin = gbs.drop('sum', axis=1)
    count_cumulated_margin = list(gbs['sum'])
    return unique_data_margin, count_cumulated_margin

def margprob(data, counts = None, margin = None, MC = 1000, xi = None):
    if counts is None:
        counts = np.repeat(1, data.shape[0])
    if margin is None:
        data_margin = data
    else:
        if type(margin) == str:
            margin = [margin]
        data_margin = data[margin]
    if counts is not None:
        if MC > 0:
            unique_data_margin = data_margin.drop_duplicates()
            response = np.repeat(0, len(counts))
            data_prob = np.array(counts)/sum(counts)
            cumul_data_prob = np.cumsum(data_prob)
            count_cumulated_margin = 0
            if xi is None:
                xi = np.random.rand(MC)
            for it in xi:                 
                whichi = it <= cumul_data_prob
                realization_data_idx = whichi.tolist().index(True)
                response[realization_data_idx] = response[realization_data_idx] + 1
            observed_data_margin = data_margin.iloc[np.where(response > 0)[0],]
            cols = data_margin.columns
            temp_df = data_margin.copy()
            temp_df['counts'] = list(response)
            gb = temp_df.groupby(list(cols), observed=True)
            gbs = gb.counts.sum().reset_index(name='sum')
            unique_data_margin = gbs.drop('sum', axis=1)
            count_cumulated_margin = list(gbs['sum'])
        else:
            unique_data_margin, count_cumulated_margin = margprob_old(data, counts, margin)
    return unique_data_margin, count_cumulated_margin
