import pandas as pd

from BN_structure_learning import learn_tbnl_graph_from_counts, write_xdsl_file

if __name__ == '__main__':

    max_features = 3
    constraints = {}
    constraints['max_features'] = max_features
    constraints['rest_max_features'] = max_features
    constraints['min_info_gain'] = 0.01
    constraints['rest_min_info_gain'] = 0.05
    filename = '/Users/aviv.gruber/dev/TBNL/Dataset_filtered for case 1_sample_1k.csv'
    df = pd.read_csv(filename)
    target = "wh_OI_bound_ratio"
    columns_to_remove = ["id"]
    columns_to_process = []
    for column in df.columns:
        if column != target:
            if column not in columns_to_remove:
                if df[column].nunique() < 2:
                    continue
                elif df[column].nunique() > 6:
                    col_name = f"{column}_quantiled"
                    df[col_name] = pd.qcut(df[column], q=4, duplicates='drop', precision=0).map(lambda x: x.right)
                    if f"{column}_quantiled" not in columns_to_remove:
                        columns_to_process.append(col_name)
                else:
                    col_name = column
                    columns_to_process.append(col_name)
        else:
            continue

    columns_to_process+=[target]
    
    print(f'''*******************************
******** Starting TBNL ********
******************************''')
    print('Target = ', target)
    # Learn only parents
    _, graph = learn_tbnl_graph_from_counts(data=df[columns_to_process], counts=None, constraints=constraints, target=target, family="nuclear", MC=0)
    features = list(graph.predecessors(target))
    # nuclear family to nuclear power
    # Learn grandparents
    for feature in features:
        features, graph = learn_tbnl_graph_from_counts(data=df[columns_to_process], counts=None, constraints=constraints, target=feature, family="nuclear", G=graph, MC=0)
    write_xdsl_file(dag=graph, data=df, target=target, filename= target+" "+str(max_features)+ f" nuclear power.xdsl")