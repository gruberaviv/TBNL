import bnlearn as bn
import pandas as pd


def bn_predict(bn_model, data, variables):
    preds = bn.predict(bn_model, data , variables, method=None)
    return preds


def extract_probas_from_preds(preds, val=1):
    return preds.apply(lambda x: x['p'][val], axis=1)


def prepare_evidence_for_inference(row):
    return row[row.notna()].to_dict()


def inference_bn(bn_model, variables, evidence, variables_value=[True]):
    query = bn.inference.fit(bn_model, variables, evidence)
    bq = bn.query2df(query)
    return bq