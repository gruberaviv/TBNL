import bnlearn as bn
import pandas as pd


def param_learning(dag, data):
    raw = pd.DataFrame.from_dict(dag.edges).rename(columns={0:'source', 1:'target'})
    raw['weight'] = 1
    adjmat = bn.vec2df(raw['source'], raw['target'], raw['weight'])
    DAG = bn.make_DAG(list(zip(raw['source'], raw['target'])), verbose=0)
    bn_model = bn.parameter_learning.fit(DAG, data[list(DAG['adjmat'].columns)], verbose=3)
    return bn_model