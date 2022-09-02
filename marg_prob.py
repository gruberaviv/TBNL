# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:45:36 2018

@author: a00408194
"""

#import tqdm

from os import name


def rows_in_a1_that_are_in_a2(a1,a2):
    
    a1_vec = list()
    a2_vec = list()
    a1_with_a2_rows = list()
    
    for a1_ind in a1.index.tolist():
        jo = a1.loc[a1_ind].tolist()
        if isinstance(jo, list):
            a1_vec.append('_'.join(str(x) for x in jo))
        else:
            a1_vec.append('_'.join(str(x) for x in [jo]))
        
    for a2_ind in a2.index.tolist():
        jo = a2.loc[a2_ind].tolist()
        if isinstance(jo, list):
            a2_vec.append('_'.join(str(x) for x in jo))
        else:
            a2_vec.append('_'.join(str(x) for x in [jo]))
    
    a1_with_a2_rows = list()
    i=-1
    for x in a1_vec:
        i=i+1
        x_in_a2 = [s for s in a2_vec if x in s]
        if len(x_in_a2) > 0:
                a1_with_a2_rows.append(i)
    
    return a1_with_a2_rows

def margprob_old(data, counts = None, margin = None):
    import numpy as np
    import pandas as pd
    
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
                      
    
    # if len(data_margin.shape) == 1:
    #     data_margin = pd.DataFrame([data_margin]).T

    #unique_data_margin = data_margin.copy().drop_duplicates()
    


    cols = data_margin.columns
    temp_df = data_margin.copy()
    #unique_data_margin = temp_df.drop_duplicates()
    temp_df['counts'] = counts
    #temp_df['combined'] = temp_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    #temp_df = temp_df[['combined', 'counts']]
    #data_margin['counts'] = counts
    #gb = temp_df.groupby('combined')
    gb = temp_df.groupby(list(cols), observed=True)
    gbs = gb.counts.sum().reset_index(name='sum')
    #del gbs['index']
    unique_data_margin = gbs.drop('sum', axis=1)
    count_cumulated_margin = list(gbs['sum'])

    # if len(unique_data_margin.shape) == 1:
    # unique_data_margin.columns = margin


#        count_cumulated_margin = list()

#        for x in unique_data_margin.index.tolist():
#            count_cumulated_margin.append(np.asarray(counts)[rows_in_a1_that_are_in_a2(data_margin, unique_data_margin.loc[x:x])].sum())

    return unique_data_margin, count_cumulated_margin

def margprob(data, counts = None, margin = None, MC = 1000, xi = None):
    
    import numpy as np
    import pandas as pd
  
    if counts is None:
        counts = np.repeat(1, data.shape[0])
  #if (is.null(counts) == TRUE)
   # counts <- rep(1,dim(data)[1])

    if margin is None:
        data_margin = data
    else:
        if type(margin) == str:
            margin = [margin]
        data_margin = data[margin]
                
        
    # if len(data_margin.shape) == 1:
    #     data_margin = pd.DataFrame([data_margin]).T
                      
    if counts is not None:
        if MC > 0:
            unique_data_margin = data_margin.drop_duplicates()
            response = np.repeat(0, len(counts))
            data_prob = np.array(counts)/sum(counts)
            cumul_data_prob = np.cumsum(data_prob)
    #     
            count_cumulated_margin = 0
    #     
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

            # observed_count = response[response>0]

            # count_cumulated_margin = list()
 
            # for x in unique_data_margin.index.tolist():
            #     count_cumulated_margin.append(observed_count[rows_in_a1_that_are_in_a2(observed_data_margin, unique_data_margin.loc[x:x,])].sum())

        else:
            unique_data_margin, count_cumulated_margin = margprob_old(data, counts, margin)
        
    return unique_data_margin, count_cumulated_margin

def margprob_orig(data, counts = None, margin = None):
    import numpy as np
    import pandas as pd
    
    if counts is None:
        counts = np.repeat(1, data.shape[0])

    if margin is None:
        data_margin = data
    else:
        if len(margin) > 0:
            data_margin = data[margin]
        else:
            data_margin = data
                      
    if len(data_margin.shape) == 1:
        data_margin = pd.DataFrame([data_margin]).T

    unique_data_margin = data_margin.drop_duplicates()
    
    if len(unique_data_margin.shape) == 1:
        unique_data_margin.columns = margin

    count_cumulated_margin = list()
    for x in unique_data_margin.index.tolist():
        count_cumulated_margin.append(np.asarray(counts)[rows_in_a1_that_are_in_a2(data_margin, unique_data_margin.loc[x:x])].sum())
        
    return unique_data_margin, count_cumulated_margin
