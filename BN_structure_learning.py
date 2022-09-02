# -*- coding: utf-8 -*-
import pickle as pkl
from feature_selection_from_counts import feature_selection_from_counts
from marg_prob import margprob, rows_in_a1_that_are_in_a2
import networkx as nx
import pandas as pd
from entropy_from_counts import flatten_list
import numpy as np
from discretize_by_MI import discretize_by_mi, discretize
import os
from itertools import product

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pkl.dump(obj, output)

def load_object(filename):
    with open(filename, 'rb') as obj:
        return pkl.load(obj)


def learn_tbnl_graph_from_counts(data, counts, target, constraints = None, family = "nuclear", G = None, MC = 0, visited_set=None):
    
    local_constraints = constraints.copy()

    if visited_set is None:
        visited_set = set()

    #print("\nPreprocessing", target,"...")
    unique_data, unique_count = margprob(data, counts, None, MC)

    if G is None:
        G = nx.DiGraph()
        G.add_nodes_from(list(data))
        G.nodes[target]['color'] = 'orange'
        G.nodes[target]['value'] = 400

    features, graph = feature_selection_from_counts(unique_data, unique_count, target, local_constraints, G, MC)
    visited_set.add(target)
    
    for entry in local_constraints.keys():
        if entry.startswith('rest_'):
            exec('local_constraints[entry.split("rest_")[1]]=local_constraints[entry]')

    if family == "nuclear":
        if len(features) > 0:
            nuclear_data = unique_data[features]
            for feature in features:
                if not feature in visited_set:
                    features, graph = learn_tbnl_graph_from_counts(nuclear_data, unique_count, feature, constraints, family, graph, MC, visited_set)
    else:
        IG_increment = local_constraints.get('extended_IG_increment', 0.0)
        local_constraints['rest_min_info_gain'] += IG_increment
 
        for feature in features:
            if not feature in visited_set:
                features, graph = learn_tbnl_graph_from_counts(unique_data, unique_count, feature, local_constraints, family, graph, MC, visited_set)

    return features, graph

def write_xdsl_file(dag, data, target, filename=None):

    print('\nStarting parameter learning...')
    
    output2File1 = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n<smile version=\"1.0\" id=\"Network2\" numsamples=\"1234\">\n<nodes>"
    output2File2 = "\n</nodes>\n<extensions>\n<genie version=\"1.0\" app=\"GeNIe 2.0.3092.0\" name=\"Network2\" faultnameformat=\"nodestate\">"

    printed, output2File1, output2File2 = printNode(dag, data, target, True, output2File1, output2File2, printed=[])
    
    output2File2 = output2File2 + "</genie>\n</extensions>\n</smile>"

    output2File1 = output2File1 + output2File2
    
    with open(filename, "w") as text_file:
        text_file.write(output2File1)

def printNode(dag, data, node, is_target, output2File1, output2File2, printed=[], pos_x=10, pos_y=600):

    print('Printing node', node)
    if node not in printed:
        parents = list(dag.predecessors(node))
        print('parents:', parents)
        for parent in parents:
            if parent not in printed:
                pos_x+=200
                printed, output2File1, output2File2 = printNode(dag, data, parent, False, output2File1, output2File2, printed, pos_x, pos_y-100)
    
    print('Printing probability table for', node)
    output2File1 = output2File1 + "\n<cpt id=\"" + node + "\">"
    states, conti = margprob(data[node], MC=0)
    states = flatten_list(states.values.tolist())
    for state in states:
        output2File1 = output2File1 + "\n<state id=\"s" + str(state).replace('.', '_') + "\"/>"
    
    output2File1 = output2File1 + "\n<parents>"
    
    parents = list(dag.predecessors(node))
    for parent in parents:
        output2File1 = output2File1 + parent + " "
    
    output2File1 = output2File1 + "</parents>\n<probabilities>"
    
    if parents == []:
        prob = [float(x)/sum(conti) for x in conti]
    else:
        full_parents_domain = obtain_joint_full_domain(data[parents]).sort_values(parents)
        prob = []
        for cond in full_parents_domain.iterrows():
            candi = pd.DataFrame(data=[cond[1]])
            indi = rows_in_a1_that_are_in_a2(data[parents], candi)
            vals, counts = margprob(data.loc[indi,:], margin=node, MC=0)
            vals = flatten_list(vals.values.tolist())
            set_vals = set(vals)
            set_states = set(states)
            #if set_vals < set_states:
            corrected_counts = np.repeat(1, len(states))
            contained = set_states.intersection(set_vals)

            for cc in contained:
                corrected_counts[states.index(cc)] += counts[vals.index(cc)]
            ccl = corrected_counts.tolist()
            probab = [float(x)/sum(ccl) for x in ccl]
            prob.append(probab)
        
    probs = str(flatten_list(prob))
    probs = probs.replace(',', "")
    probs = probs.replace('[', "")
    probs = probs.replace(']', "")
    output2File3 = probs
 
    output2File1 = output2File1 + output2File3
    
    output2File1 = output2File1 + "</probabilities>\n</cpt>"
    
    if is_target:
        output2File2 = output2File2 + "<node id=\"" + node + "\">\n<name>" + node + "</name>\n<interior color=\"ffff99\" />" + \
                    "\n<outline color=\"000080\" />\n<font color=\"000000\" name=\"Arial\" size=\"11\"  bold=\"true\" />\n<position>"

    else:
        output2File2 = output2File2 + "<node id=\"" + node + "\">\n<name>" + node + "</name>\n<interior color=\"e5f6f7\" />" + \
                    "\n<outline color=\"000080\" />\n<font color=\"000000\" name=\"Arial\" size=\"11\"  bold=\"true\" />\n<position>"
    
    output2File2 = output2File2 + str(pos_x) + " " + str(pos_y) + " " + str(pos_x + 80) + " " + str(pos_y + 30)
                                 
    output2File2 = output2File2 + "</position>\n</node>"
        
    printed.append(node)
    return printed, output2File1, output2File2

def obtain_joint_full_domain(data):

    if isinstance(data, dict):
        var_states = data
        my_vars = list(data.keys())
    else:
        my_vars = list(data.columns)
        var_states = []
        for my_var in my_vars:
            temp_states, _ = margprob(data[my_var], MC = 0)
            var_states.append((my_var, flatten_list(temp_states.values.tolist())))
    
        var_states = dict(var_states)
    
    parents_unique_values = []
    for par in my_vars:
        parents_unique_values.append(list(data[par].unique()))
    mat = pd.DataFrame(list(product(*parents_unique_values)), columns=my_vars)
    return mat

def sliding_tbnl(data, target, constraints, rolling_time_window_in_days = 30, rolling_jumps_in_days = 7, use_saved_graphs = False):
    

    df = data[['START_TIME', 'END_TIME', 'mfc_id', target]].copy()

    if df[target].nunique() > 8:
        try:
            tar = target + '_qdisc'
            df[tar] = discretize(df[target], bins=[0, .25, .5, .75, 1.], range_or_quantile='quantile').apply(lambda x: x.right)
        except Exception:
            print('Failed to discretize', target, 'by equi-probability.')
            try:
                tar = target + '_rdisc'
                df[tar] = discretize(df[target], range_or_quantile='range').apply(lambda x: x.right)
            except Exception:
                print('Failed to discretize', target, 'by equi-range.')
                tar = target
    else:
        tar = target

    filename = 'FeatureGenerator_disc_about_'+tar+'_in_'+df.iloc[0]['mfc_id']+'.feather'
    if os.path.exists(filename):
        df = pd.read_feather(filename)
    else:
        df_numeric = data.copy().sort_values('START_TIME').reset_index(drop=True)
        df_numeric = df_numeric.select_dtypes(include=[np.number])

        for col in df_numeric.columns:
            if col != target:
                if df_numeric[col].nunique() > 6:
                    try:
                        print('Trying to discretize', col, 'about', tar)
                        if df[tar].nunique() > 1:
                            df[col+'_midisc'] = np.round(discretize_by_mi(df_numeric[col], df[tar], max_values=10),0)
                        else:
                            print('No model built for', tar, 'in mfc', df.iloc[0]['mfc_id'], 'since it is always', df.iloc[0][tar])
                            continue
                    except:
                        print("Can't discretize", col, "by MI about", tar)
                else:
                    df[col] = df_numeric[col]

        df.reset_index(drop=True).to_feather(filename)

    min_start_time = df['START_TIME'].min()
    max_start_time = df['START_TIME'].max()

    num_of_time_slices = interval_duration(portion.closed(min_start_time, max_start_time))//(rolling_jumps_in_days*24*3600) + 1

    for time_slice_idx in range(np.int(num_of_time_slices)):
        curr_start_time = min_start_time + pd.to_timedelta(time_slice_idx*rolling_jumps_in_days, unit='d')
        curr_end_time = curr_start_time + pd.to_timedelta(rolling_time_window_in_days, unit='d')

        mask = (df['START_TIME'] > curr_start_time) & (df['START_TIME'] <= curr_end_time)

        df_slice = df.loc[mask]

        print('Working on', df.iloc[0]['mfc_id'], df_slice['START_TIME'].min(), '-', df_slice['END_TIME'].max())

        if df_slice[tar].nunique() < 2:
            continue

        df_slice = df_slice[(df_slice.columns[df_slice.apply('nunique') < 8]) & (df_slice.columns[df_slice.apply('nunique') > 1])]
        
        if df_slice.shape[0]*df_slice.shape[1] == 0:
            continue

        print('''*******************************
******** Starting TBNL ********
*******************************''')
        print('Target = ', tar)

        path_to_file = tar+" "+df.iloc[0]['mfc_id']+" "+str(curr_start_time.date())+' for '+str(rolling_time_window_in_days)+' days'

        if use_saved_graphs:
            try:
                graph = load_object(path_to_file+".pkl")
            except:
                print('Graph file not found! Try not using saved (pickle) files')
                continue
        else:
            _, graph = learn_tbnl_graph_from_counts(data=df_slice, counts=None, constraints=constraints, target=tar, family="nuclear", MC=1000)

            features = list(graph.predecessors(tar))
            for feature in features:
                features, graph = learn_tbnl_graph_from_counts(data=df_slice, counts=None, constraints=constraints, \
                    target=feature, family="nuclear", G=graph, MC=1000)

            save_object(graph, path_to_file+".pkl")
        

        write_xdsl_file(dag=graph, data=df_slice, target=tar, filename=path_to_file+".xdsl")
#=============================================================================
