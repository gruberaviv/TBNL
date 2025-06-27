from pickletools import TAKEN_FROM_ARGUMENT4
import bnlearn as bn
import pandas as pd
import numpy as np

def param_learning(dag, data):
    raw = pd.DataFrame.from_dict(dag.edges).rename(columns={0:'source', 1:'target'})
    raw['weight'] = 1
    adjmat = bn.vec2df(raw['source'], raw['target'], raw['weight'])
    DAG = bn.make_DAG(list(zip(raw['source'], raw['target'])), verbose=0)
    bn_model = bn.parameter_learning.fit(DAG, data[list(DAG['adjmat'].columns)], verbose=3)
    return bn_model

def discretize_with_na(df, colname, quantiles=4):
    series_notna = df[df[colname]!='na'][colname].astype(float)
    a = pd.qcut(series_notna,
    q=quantiles, duplicates='drop', precision=0)

    categories = pd.IntervalIndex.from_tuples([(-0.001, a.cat.categories[0].right)] + 
                                    [(a.cat.categories[i].left, a.cat.categories[i].right) for i in range(1, len(a.cat.categories))])

    # a.cat.add_categories(pd.interval_range(start=-1, end=0, closed='right'))
    # categories = a.cat.categories
    a = pd.cut(series_notna, bins=categories)
    b = df[df[colname]=='na']
    b[colname] = np.repeat(pd.interval_range(start=-1, end=-0.001, freq=0.999, closed='right'), repeats=len(b))
    b = b[colname]
    c = pd.concat([a,b])
    return c.sort_index()

def replace_str_with_na_to_float(df, colname):
    a = df[df[colname]!='na'][colname].astype(float)
    b = df[df[colname]=='na']
    b[colname] = -0.5
    b = b[colname]
    return pd.concat([a,b]).sort_index()

def transform_values_to_intervals(df, variable):
    intervals_array = df[variable].unique()
    intervals_filtered = [i for i in intervals_array if pd.notnull(i)]
    intervals_index = pd.IntervalIndex(intervals_filtered)
    return intervals_index
    

def prep_case(df, case, dag_case):
    from sklearn.model_selection import train_test_split
    df_case = df[df['Cases']==case]
    train_case, test_case = train_test_split(df_case, test_size=0.3)

    train_case.reset_index(drop=True, inplace=True)
    test_case.reset_index(drop=True, inplace=True)

    train_case['R^(x_b;x_w)'] = discretize_with_na(train_case, 'R^(x_b;x_w)', quantiles=10)
    train_case['R^(y_r;x_w)'] = discretize_with_na(train_case, 'R^(y_r;x_w)', quantiles=10)
    df_train_discretized = train_case.apply(lambda c: pd.cut(c, bins=10) if type(c.iloc[0]) in (np.int64, np.float64) else c)
    tbnl_case = bn.parameter_learning.fit(dag_case, df_train_discretized[list(dag_case['adjmat'].columns)], verbose=3)

    nodes_states = {}
    for cpd in tbnl_case['model'].cpds:
        if hasattr(cpd, 'state_names'):
            nodes_states[cpd.variable] = cpd.state_names[cpd.variable]
        else:
            nodes_states[cpd.variable] = 'unknown'

    test_case['R^(x_b;x_w)'] = replace_str_with_na_to_float(test_case, 'R^(x_b;x_w)')
    intervals_index = transform_values_to_intervals(train_case,'R^(x_b;x_w)')
    test_case['R^(x_b;x_w)'] = pd.cut(test_case['R^(x_b;x_w)'], intervals_index)

    test_case['R^(y_r;x_w)'] = replace_str_with_na_to_float(test_case, 'R^(y_r;x_w)')
    intervals_index = transform_values_to_intervals(train_case,'R^(y_r;x_w)')
    test_case['R^(y_r;x_w)'] = pd.cut(test_case['R^(y_r;x_w)'], intervals_index)
    
    df_test_discretized = test_case.copy()
    for col in df_test_discretized.columns:
        print(col)
        if col in list(nodes_states.keys()):
            if type(test_case.iloc[0][col]) in (np.int64, np.float64):
                intervals_index = transform_values_to_intervals(df_train_discretized, col)
                df_test_discretized[col] = pd.cut(df_test_discretized[col], intervals_index)
    #df_test_discretized = test_case.apply(lambda c: pd.cut(c, nodes_states[c.name]) if (type(c.iloc[0]) in (np.int64, np.float64)) & (c.name in list(nodes_states.keys())) else c)

    return tbnl_case, df_train_discretized, df_test_discretized


def test_model(model, test_set, label):

    X_test_bn = test_set.drop(label, axis=1)
        # Inference
    # Initialize a list to store the probabilities
    probabilities_list = []
    # Initialize a list to store the target group with the highest probability for each row
    highest_prob_targets = []
    model_keys = list(model['model'].nodes())
    # Iterate over each row in the test DataFrame
    for index, row in X_test_bn.iterrows():
        # Construct evidence dictionary from the row
        # evidence = row.to_dict()
        evidence = {var: row[var] for var in X_test_bn.columns if ((var != label) and (var != 'Highest_Prob_Target') and \
                    (var in model_keys))}

        # If you want to exclude the target variable ('Duration_Group_30' in this case) from the evidence, do it here
        # evidence.pop('Duration_Group_30', None)  # Remove the target variable if it's in the dataframe

        try:
            #Perform the query
            #query_result = bn.inference.fit(model_update, variables=[target], evidence=evidence)
            #inference_bn(my_bn, variables=[target], evidence=evidence)
            query_result = inference_bn(model, variables=[label],evidence=evidence)
            # Extract the probability distribution for 'Duration_Group_30'
            highest_index = query_result['p'].idxmax()
            highest_string_value_direct = query_result.loc[highest_index, label]
            # Append to the list
            highest_prob_targets.append(highest_string_value_direct)
        except Exception as e:  # I think that there are errors because of evidence groups that don't happen in the train dataset
            print(f"Error with initial evidence: {e}")
            # In case of error, try removing problematic evidence one by one
            evidence_keys = list(evidence.keys())  # Get all keys to a list
            total_keys = len(evidence_keys)  # Total number of keys
            for index_ev, var_to_remove in enumerate(evidence_keys):
                # for var_to_remove in list(evidence.keys()):
                try:
                    # Create a copy of the evidence and remove one variable
                    evidence_attempt = copy.deepcopy(evidence)
                    evidence_attempt.pop(var_to_remove, None)  # Remove potentially problematic variable
                    # Retry the query with the modified evidence
                    query_result = inference_bn(tbnl_case1, variables=[label], evidence=evidence_attempt)
                    print(f"Query successful after removing: {var_to_remove}")
                    # Extract the probability distribution for 'Duration_Group_30'
                    highest_index = query_result['p'].idxmax()
                    highest_string_value_direct = query_result.loc[highest_index, label]
                    # Append to the list
                    highest_prob_targets.append(highest_string_value_direct)
                    break  # Exit the loop if successful
                except Exception as e:
                    # If it still fails, continue trying by removing the next variable
                    if index_ev == total_keys - 1:
                        print(index)
                        print(row)
                        print(evidence)
                        highest_prob_targets.append('Unkown evidence')
                        break
                    continue

    X_test_bn['Highest_Prob_Target'] = highest_prob_targets
    #X_test_bn.to_csv('/Users/talkatz/Persistence model/modeling/tbnl_per_res_30min.csv', index=False)
    return X_test_bn


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from bn_inference import inference_bn
    import copy
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    import xgboost as xgb

    filename = '/Users/avivgruber/dev/TBNL/sam3/data/Dataset_to_predict.csv'
    df = pd.read_csv(filename)

    # CASE 1

    # Example: Define the DAG structure manually
    DAG_case1 = bn.make_DAG([('R^(x_b;x_w)', 'R^(y_r;x_w)'), ('Tb', 'R^(y_r;x_w)'), ('c', 'R^(y_r;x_w)'), ('S~0', 'R^(y_r;x_w)'),
    ('R^(pai1;h1)', 'c'), ('Costs_relation_1', 'c'), ('R^(c;z)', 'c'),
    ('c', 'R^(x_b;x_w)'), ('pai_2', 'R^(x_b;x_w)'), ('S~2', 'R^(x_b;x_w)'),
    ('R^(x_b;x_w)', 'S~0'), ('S~1', 'S~0'),
    ('Tr_2', 'Tb'),
    ('Costs_relation_1', 'R^(pai1;h1)'),
    ('Costs_relation_1', 'R^(c;z)'),
    ('R^(x_b;x_w)', 'S~1')])

    accuracy = {}
    case_parames = {}
    tbnl_case1, train_case1, test_case1 = prep_case(df, 1, DAG_case1)
    target = 'R^(y_r;x_w)'
    res = test_model(tbnl_case1, test_case1.drop('R^(x_b;x_w)', axis=1), target)
    nodes_states = {}
    for cpd in tbnl_case1['model'].cpds:
        if hasattr(cpd, 'state_names'):
            nodes_states[cpd.variable] = cpd.state_names[cpd.variable]
        else:
            nodes_states[cpd.variable] = 'unknown'


    cm = confusion_matrix(y_true=test_case1[test_case1[target].notna()][target].astype(str), y_pred=res['Highest_Prob_Target'].astype(str))
    accuracy['TBNL'] = np.trace(cm) / np.sum(cm).astype('float')
    # fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nodes_states[target], )
    # ax.set(title='Accuracy='+str(round(accuracy, 2)))
    # disp.plot(ax=ax)

    X_train = train_case1.drop(target, axis=1)
    y_train = train_case1[target]
    X_test = test_case1.drop(target, axis=1)
    y_test = test_case1[target]

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from sklearn.impute import SimpleImputer


    # Sample DataFrame
    # df = pd.DataFrame(data) # Assuming 'data' is your dataset

    # Identify categorical and numerical columns
    categorical_cols = train_case1.select_dtypes(include=['object', 'category']).columns
    numerical_cols = train_case1.select_dtypes(include=['int64', 'float64']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    # Define the models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Create pipelines for each model
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', rf_model)])
    dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', dt_model)])
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', xgb_model)])

    # Train and evaluate each model
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    for name, pipeline in [('Random Forest', rf_pipeline), ('Decision Tree', dt_pipeline), ('XGBoost', xgb_pipeline)]:
        pipeline.fit(X_train, y_train_encoded)
        predictions = pipeline.predict(X_test.drop('R^(x_b;x_w)', axis=1))
        decoded_labels = label_encoder.inverse_transform(predictions)
        accuracy[name] = accuracy_score(y_test_encoded, predictions)
    
    print('Accuracy:', accuracy)


    avgdaf=231
    
