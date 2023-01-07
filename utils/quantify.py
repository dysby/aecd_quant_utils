import quapy as qp
from quapy.data import LabelledCollection
import quapy.functional as F
import pandas as pd


def generate_tests_by_month(all_data) -> LabelledCollection:
    for month in range(5,8):
        test_split_data = all_data[all_data.month == month].drop(['fraud_bool','month'], axis=1)
        test_split_labels = all_data[all_data.month == month].fraud_bool
        yield qp.data.LabelledCollection(test_split_data, test_split_labels, classes_=[0,1])


def test_quantifier(quantifier_list, transformed_data):
    tests = []

    for i, df_test in enumerate(generate_tests_by_month(transformed_data)):
        
        for (idx, calibration_method, q_method, quantifier) in quantifier_list:

          estim_prevalence = quantifier.quantify(df_test.instances)
          true_prevalence  = df_test.prevalence()
          error = qp.error.mae(true_prevalence, estim_prevalence)

          tests.append({
              'Month': 5+i,
              'Model': idx,
              'Quantifier': q_method,
              'Calibration': calibration_method,
              'Mean Absolute Error (MAE)': error,
              'Prevalence (Estimated)': estim_prevalence[1],
              'Prevalence (True)':true_prevalence[1],
          })

    return pd.DataFrame(tests)


def generate_quantifiers(clfs, dataset_training, dataset_eval):
    quantifiers = []

    for idx, calibration_method, clf, thr in clfs:
        #qname = name + "_CCt"
        q_method = 'CCt'
        q = qp.method.custom_threshold.CCWithThreshold(clf, thr)
        q.fit(dataset_training, fit_learner=False)
        quantifiers.append((idx, calibration_method, q_method, q))

        #qname = name + "_ACCt"
        q_method = 'ACCt'
        q = qp.method.custom_threshold.ACCWithThreshold(clf, thr)
        q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        quantifiers.append((idx, calibration_method, q_method, q))

        #qname = name + "_PACC"
        q_method = 'PACC'
        q = qp.method.aggregative.PACC(clf)
        q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        quantifiers.append((idx, calibration_method, q_method, q))

        #qname = name + "_SLD"
        q_method = 'SLD'
        q = qp.method.aggregative.SLD(clf)
        q.fit(dataset_training, fit_learner=False)
        quantifiers.append((idx, calibration_method, q_method, q))

        #qname = name + "_HDy"
        q_method = 'HDy'
        q = qp.method.aggregative.HDy(clf)
        q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        quantifiers.append((idx, calibration_method, q_method, q))

        #qname = name + "_MS"
        #q = qp.method.aggregative.MS(clf)
        #q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        #quantifiers.append((qname, q))

    return quantifiers
