import quapy as qp
import pandas as pd

from utils.data import generate_tests_by_month

def test_quantifiers(quantifier_list, transformed_data):
    tests = []

    for i, df_test in enumerate(generate_tests_by_month(transformed_data)):

        for (idx, calibration_method, q_method, quantifier) in quantifier_list:

            estim_prevalence = quantifier.quantify(df_test.instances)
            true_prevalence = df_test.prevalence()
            error = qp.error.mae(true_prevalence, estim_prevalence)
            mrae = qp.error.mrae(true_prevalence, estim_prevalence)
            mkld = qp.error.mkld(true_prevalence, estim_prevalence)

        tests.append(
            {
                "Month": 5 + i,
                "Model": idx,
                "Quantifier": q_method,
                "Calibration": calibration_method,
                "MAE": error,
                "MRAE": mrae,
                "KLD": mkld,
                "Prevalence (Estimated)": estim_prevalence[1],
                "Prevalence (True)": true_prevalence[1],
            }
        )

    return pd.DataFrame(tests)


def generate_quantifiers(clfs, dataset_training, dataset_eval):
    quantifiers = []

    for idx, calibration_method, clf, thr in clfs:
        # qname = name + "_CCt"
        q_method = "CC"
        q = qp.method.aggregative.CC(clf)
        q.fit(dataset_training, fit_learner=False)
        quantifiers.append((idx, calibration_method, q_method, q))

        q_method = "CCt"
        q = qp.method.custom_threshold.CCWithThreshold(clf, thr)
        q.fit(dataset_training, fit_learner=False)
        quantifiers.append((idx, calibration_method, q_method, q))

        # qname = name + "_ACCt"
        q_method = "ACCt"
        q = qp.method.custom_threshold.ACCWithThreshold(clf, thr)
        q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        quantifiers.append((idx, calibration_method, q_method, q))

        # qname = name + "_PACC"
        q_method = "PACC"
        q = qp.method.aggregative.PACC(clf)
        q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        quantifiers.append((idx, calibration_method, q_method, q))

        # qname = name + "_SLD"
        q_method = "SLD"
        q = qp.method.aggregative.SLD(clf)
        q.fit(dataset_training, fit_learner=False)
        quantifiers.append((idx, calibration_method, q_method, q))

        # qname = name + "_HDy"
        q_method = "HDy"
        q = qp.method.aggregative.HDy(clf)
        q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        quantifiers.append((idx, calibration_method, q_method, q))

        # qname = name + "_MS"
        # q = qp.method.aggregative.MS(clf)
        # q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        # quantifiers.append((qname, q))

    return quantifiers
