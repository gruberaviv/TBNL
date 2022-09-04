# TBNL
## Targeted Bayesian Network Learning
### The TBNL builds a BN classifier and saves the resulted network onto a file which can be viewed by GeNIe framework.
### Generally speaking, Bayesian network learning has two learning stages: structure learning (the graph) and parameter learning (the probability distributions).
### GeNIe is a third party app. There are several ways for licensing GeNIe and this is on the user's responsibility.
### Using GeNIe is not mandatory, but then the user would have to modify the code to their needs, as the parameter learning stage is tailored for GeNIe.
### The special value of the TBNL is that it places the target node at the bottom of the network, and gradually selects the most influential parents of it, by maximizing the total information about it, constrained by graphical constraints and informational constraints.
### The same routine is then applied to each one of the parents in turn, resulting in a quasi-cuasal explanatory classifier.
#### GeNIe: https://www.bayesfusion.com/genie/
#### TBNL: Gruber, A., & Ben-Gal, I. (2019). A targeted Bayesian network learning for classification. Quality Technology and Quantitative Management, 16(3), 243-261. https://doi.org/10.1080/16843703.2017.1395109

Here is a setup which can be run (only modify the data reading to a specified file, and predetermine the target variable)
import pandas as pd
from BN_structure_learning import learn_tbnl_graph_from_counts, write_xdsl_file
max_features = 3
constraints = {}
constraints['max_features'] = max_features
constraints['rest_max_features'] = max_features
constraints['min_info_gain'] = 0.01
constraints['rest_min_info_gain'] = 0.05
filename = 'path_to_file.csv'
df = pd.read_csv(filename)
target = "target colname"
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

Footer
