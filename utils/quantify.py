import quapy as qp
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from utils.data import generate_tests_by_month



def test_quantifiers(quantifier_list, transformed_data):
    tests = []

    for m, df_test in enumerate(generate_tests_by_month(transformed_data)):

        true_prevalence = df_test.prevalence()

        for (idx, calibration_method, q_method, quantifier) in quantifier_list:
            
            estim_prevalence = quantifier.quantify(df_test.instances)
            error = qp.error.mae(true_prevalence, estim_prevalence)
            mrae = qp.error.mrae(true_prevalence, estim_prevalence)
            mkld = qp.error.mkld(true_prevalence, estim_prevalence)

            tests.append(
                {
                    "Month": 6 + m,
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

def test_N_samples_from_month(quantifier_list, transformed_data, N=10, fraction=0.7):
    tests = []

    for m, df_test in enumerate(generate_tests_by_month(transformed_data)):

        true_prevalence = df_test.prevalence()
        
        X_test = df_test.instances
        y_test = df_test.labels

        for (idx, calibration_method, q_method, quantifier) in quantifier_list:

            estim_prevalence = quantifier.quantify(X_test)
            error = qp.error.mae(true_prevalence, estim_prevalence)
            mrae = qp.error.mrae(true_prevalence, estim_prevalence)
            mkld = qp.error.mkld(true_prevalence, estim_prevalence)

            tests.append(
                {
                    "Month": 6 + m,
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


def do_n_tests_with_subsampling_on_qlist(qs, X_test: pd.DataFrame, y_test: pd.DataFrame, N=10, fraction=0.8):
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

    sss = StratifiedShuffleSplit(n_splits=N, test_size=fraction, random_state=42)

    results = []

    # for i, (train_index, test_index) in enumerate(sss.split(X_test, y_test)):
    for _, test_index in sss.split(X_test, y_test):
        #print(f"Fold {i}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")

        p = y_test[test_index].sum() / len(y_test[test_index]) 
        true_prevalence = [1 - p, p]

        for (idx, calibration_method, q_name, q) in qs:
            estim_prevalence = q.quantify(X_test[test_index])
            ae = qp.error.ae(true_prevalence, estim_prevalence)
            rae = qp.error.rae(true_prevalence, estim_prevalence)
            kld = qp.error.kld(true_prevalence, estim_prevalence)
            
            results.append(
                {
                    "Model": idx,
                    "Quantifier": q_name,
                    "Calibration": calibration_method,
                    "MAE": ae,
                    "MRAE": rae,
                    "KLD": kld,
                    "Prevalence (Estimated)": estim_prevalence[1],
                    "Prevalence (True)": true_prevalence[1],
                }
            )

    return pd.DataFrame(results)

