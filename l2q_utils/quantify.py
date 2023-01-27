import quapy as qp
import quapy.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import logging
  
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from .data import generate_tests_by_month, bootstrap

# QuantifierTestResult = frozentuple(QuantifierTestResult, "Month Model Quantifier Calibration MAE MRAE KLD, prevalence_estimate, prevalence_true")

def _quantify(quantifiers, X_test, y_test):
    tests = []

    true_prevalence = F.prevalence_from_labels(y_test)

    for (idx, calibration_method, q_method, quantifier) in quantifiers:

        estim_prevalence = quantifier.quantify(X_test)
        ae = qp.error.ae(true_prevalence, estim_prevalence)
        rae = qp.error.rae(true_prevalence, estim_prevalence)
        kld = qp.error.kld(true_prevalence, estim_prevalence)

        tests.append(
            {
                "Model": idx,
                "Quantifier": q_method,
                "Calibration": calibration_method,
                "MAE": ae,
                "MRAE": rae,
                "KLD": kld,
                "Prevalence (Estimated)": estim_prevalence[1],
                "Prevalence (True)": true_prevalence[1],
            }
        )
    return tests

def test_quantifiers(quantifiers, transformed_data):
    tests = []

    for m, df_test in enumerate(generate_tests_by_month(transformed_data)):
        log.info(f"Testing quantifiers on iteration m={m}.")
        results = _quantify(quantifiers, df_test.instances, df_test.labels)
        results = pd.DataFrame.from_dict(results)[["Month"]] = 6 + m
        tests = pd.concat(tests, results)

    return pd.DataFrame(tests)


def test_quantifiers_bootstraped(quantifiers, data, n_bootstraps: int = 30, fraction=1):
    tests = pd.DataFrame()

    # Month 6, 7, 8
    log.info(f"Start Testing with n={n_bootstraps} bootstraps fraction={fraction}.")

    for m in range(6, 9):
        log.info(f"Testing m={m} for {n_bootstraps} bootstraps.")
        test_data = data[data["month"] == m - 1]

        for _ in range(n_bootstraps):
            sampleX, sampley = bootstrap(test_data, fraction=fraction)

            results = _quantify(quantifiers, sampleX, sampley)
            results = pd.DataFrame.from_dict(results)[["Month"]] = m
            tests = pd.concat(tests, results)

    return tests

def do_n_tests_with_subsampling_on_qlist(
    quantifiers, X_test: pd.DataFrame, y_test: pd.DataFrame, N=10, fraction=0.7
):
    """Tests a quantifier set on N stratified subsamples with replacement of X_test.

    Parameters
    ----------
    qs : list
        Quantifiers list to test
    X_test : pandas.DataFrame
        Test set to be sampled
    true_prevalence : numpy.ndarray
        Test set true prevalence
    N : int, Default=10
        Number of tests on X_test
    fraction: float, Default=0.8
        Stratification percentage to subsample from X_test

    Returns
    -------
    pandas.DataFrame
        DataFrame containing AE and RAE for each quantifier q in the list
    """

    results = pd.DataFrame()
    log.info(f"Start Test {N} StratifiedSuffleSplit itrerations.")

    sss = StratifiedShuffleSplit(n_splits=N, test_size=fraction, random_state=42)
    for i, test_index in sss.split(X_test, y_test):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        log.info(f"{i} itreration.")
        result = _quantify(quantifiers, X_test.loc(test_index), y_test.loc(test_index))
        results = pd.concat([results, pd.DataFrame.from_dict(result)])

    return results


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

        # q_method = "ACC"
        # q = qp.method.aggregative.ACC(clf)
        # q.fit(dataset_training, fit_learner=False, val_split=dataset_eval)
        # quantifiers.append((idx, calibration_method, q_method, q))

        q_method = "PCC"
        q = qp.method.aggregative.PCC(clf)
        q.fit(dataset_training, fit_learner=False)
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



