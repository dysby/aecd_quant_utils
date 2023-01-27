import pandas as pd
import numpy as np

import quapy as qp
from quapy.data import LabelledCollection

def artificial_prevalence_sampling(data: pd.DataFrame, p=0.2, random_state=42):
    pass

def generate_synthetic_p(df, p=0.2, random_state=42):
    transformed = pd.DataFrame()

    for m in range(0, 8):
        month_f = df[df.month.eq(m) & df.fraud_bool.eq(1)]
        month_nf = df[df.month.eq(m) & df.fraud_bool.eq(0)].sample(
            n=int(month_f.fraud_bool.sum() * (1 / p - 1)), random_state=random_state
        )
        # print(month_f.shape[0], month_nf.shape[0])
        generated_month_sampling = df.iloc[month_f.index.union(month_nf.index), :]
        transformed = pd.concat([transformed, generated_month_sampling])

    return transformed


def shuffle_between_months(base_data, random_state=42):
    transformed = pd.DataFrame()
    consumed_fraud_index = np.array([])
    consumed_nonfraud_index = np.array([])

    f_index = base_data[base_data.fraud_bool.eq(1)].index
    nf_index = base_data[base_data.fraud_bool.eq(0)].index

    for m in range(8):

        # number of fraud and non_fraud in this month m
        n_f = base_data[base_data.month.eq(m)].fraud_bool.eq(1).sum()
        n_nf = base_data[base_data.month.eq(m)].fraud_bool.eq(0).sum()

        # get available indexes that have not been sampled
        remaining_index_fraud = f_index[~f_index.isin(consumed_fraud_index)]
        remaining_index_nonfraud = nf_index[~nf_index.isin(consumed_nonfraud_index)]

        # get n_f samples of non fraud rows conditioned to available indexes
        f_samples = base_data.loc[remaining_index_fraud].sample(
            n=n_f, random_state=random_state
        )
        # update month of the new samples
        f_samples["month"] = m

        # get n_nf samples of fraud rows conditioned to available indexes
        nf_samples = base_data.loc[remaining_index_nonfraud].sample(
            n=n_nf, random_state=random_state
        )
        nf_samples["month"] = m

        # store new month samples
        transformed = pd.concat([transformed, f_samples, nf_samples])

        # make sure sampled indexes are not used again
        consumed_fraud_index = transformed[transformed.fraud_bool.eq(1)].index
        consumed_nonfraud_index = transformed[transformed.fraud_bool.eq(0)].index

    return transformed


def data_split(df, size=[4, 1, 3]):
    select_train = df.month < size[0]
    select_eval = (df.month >= size[0]) & (df.month < (size[0] + size[1]))
    select_test = df.month >= (size[0] + size[1])

    X_train = df[select_train].drop(["fraud_bool", "month"], axis=1)
    X_valid = df[select_eval].drop(["fraud_bool", "month"], axis=1)
    X_test = df[select_test].drop(["fraud_bool", "month"], axis=1)

    y_train = df[select_train].fraud_bool
    y_valid = df[select_eval].fraud_bool
    y_test = df[select_test].fraud_bool

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def generate_tests_by_month(all_data) -> LabelledCollection:
    for month in range(5, 8):
        test_split_data = all_data[all_data.month == month].drop(
            ["fraud_bool", "month"], axis=1
        )
        test_split_labels = all_data[all_data.month == month].fraud_bool
        yield qp.data.LabelledCollection(
            test_split_data, test_split_labels, classes_=[0, 1]
        )


def subsample_by_p_target(data, p):
    N = data.shape[0]

    max_positives = int(data.fraud_bool.eq(1).sum())
    max_negatives = int(data.fraud_bool.eq(0).sum())

    positives = int(N * p)
    negatives = int(N * (1 - p))

    if positives > max_positives:
        positives = max_positives
        negatives = max_positives / p - max_positives
    elif negatives > max_negatives:
        negatives = max_negatives
        positives = max_negatives / (1 - p) - max_negatives

    positives = int(positives)
    negatives = int(negatives)

    # print(f"positives = {positives}, negatives = {negatives}")
    sample = pd.concat(
        [
            data[data.fraud_bool.eq(0)].drop("month", axis=1).sample(negatives),
            data[data.fraud_bool.eq(1)].drop("month", axis=1).sample(positives),
        ]
    )

    sampleX = sample.drop("fraud_bool", axis=1)
    sampley = sample[["fraud_bool"]]

    return sampleX, sampley


def bootstrap(data: pd.DataFrame, fraction: float = 0.1):
    sample = data.sample(int(data.shape[0] * fraction), replace=True)
    return sample.drop(["month", "fraud_bool"], axis=1), sample.fraud_bool


