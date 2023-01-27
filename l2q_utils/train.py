from functools import partial
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from copy import deepcopy

import lightgbm as lgb
import optuna



def select_threshold(target, predicted, max_fpr=0.01):
    """
    Find the optimal probability cutoff point for a classification
    probalilistic model conditioned to a false positive rate.

    :param target: true class
    :param predicted: classification probalility score
    :returns threshold: threshold value
    """
    df = pd.DataFrame(zip(target, predicted), columns=["target", "scores"]).sort_values(
        by="scores", ascending=False
    )

    negatives = df.target.eq(0).sum()  # sum(df.target == 0)

    df["fpr"] = df.target.eq(0).cumsum() / negatives

    thresholds = df.scores[df.fpr <= max_fpr]

    # If no threshold found for max_fpr
    if len(thresholds) == 0:
        return df.scores.head(
            1
        ).item()  # never classifify as 1, threfore will never result in false positive.
        # return np.random.uniform(0, df.scores.tail(1).item())
        # - then return mean value between max score (probability) and 1
        # return (df.scores.head(1).item() + 1) / 2

    # check if any threshold have a fpr greather than max_fpr and exclude it
    # if any(df.fpr[df.scores.isin(thresholds.tail(1))] > max_fpr):
    #    thresholds = thresholds[~thresholds.isin(df.scores[df.fpr[df.scores.isin(thresholds)] > max_fpr])]
    #    if len(thresholds) == 0:
    #        return np.random.uniform(1, df.scores.head(1).item())
    if any(df.fpr[df.scores.eq(thresholds.tail(1).item())] > max_fpr):
        return thresholds.tail(1).item() + 0.001

    return thresholds.tail(1).item()



select_threshold_5perc_fpr = partial(select_threshold, max_fpr=0.05)
select_threshold_1perc_fpr = partial(select_threshold, max_fpr=0.01)


def objective(trial, train_df, eval_df):
    """
    :param trial: optuna trial
    :param train_df: train data encapsoleted in lightgbm dataframe
    :param train_df: train data encapsoleted in lightgbm dataframe
    :returns: Recall at 5% False Positive Rate
    """

    # X_train, X_eval, X_test, y_train, y_eval, y_test = data_split(base_data)
    # lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False, categorical_feature= [7, 14, 17, 24, 26])
    # lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train, free_raw_data=False, categorical_feature= [7, 14, 17, 24, 26])
    # lgb_test = lgb.Dataset(X_test, y_test, free_raw_data=False, categorical_feature= [7, 14, 17, 24, 26])
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "device": "gpu",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ("gbdt", "goss")),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 0.2
        ),  # loguniform(1e-5, 0.2),
        "num_iterations": trial.suggest_int(
            "num_iterations", 10, 1000
        ),  # randint(10,1000),       # randomly sample numbers from 10 to 500 estimators
        "max_bin": trial.suggest_int("max_bin", 25, 255),  # randint(25, 255),
        "num_leaves": trial.suggest_int("num_leaves", 20, 500),  # randint(20, 500),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 0, 500
        ),  # randint(0, 500),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-5, 1e4
        ),  # loguniform(1e-5, 1e4),
        # 'subsample': uniform(loc=0.2, scale=0.8),
        "max_depth": trial.suggest_int("max_depth", 3, 25),  # randint(3, 25),
        # 'feature_fraction': uniform(loc=0.4, scale=0.6),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),  # uniform(0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),  # uniform(0, 10),
        #'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
        #'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1),
    }

    # Add a callback for pruning.
    # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    gbm = lgb.train(
        param,
        train_df,
        valid_sets=[eval_df],
        verbose_eval=False,  # , callbacks=[pruning_callback]
    )

    # since we use lgb.train gbm is a native lightgbm classifier
    # and gbm.predict will return probabilities, only return scores if raw_score=True
    y_scores = gbm.predict(eval_df.get_data())
    trh = select_threshold_5perc_fpr(eval_df.get_label(), y_scores)

    trial.set_user_attr("threshold_5perc_fpr", trh)

    y_pred = np.where(y_scores < trh, 0, 1)

    recall_at_5perc_fpr = recall_score(eval_df.get_label(), y_pred)

    # f1_score_at_1perc_fpr = f1_score(eval_df.get_label(), y_pred)
    balanced_accuracy = balanced_accuracy_score(eval_df.get_label(), y_pred)
    # auc = roc_auc_score(y_eval, y_score=y_scores)

    return recall_at_5perc_fpr, balanced_accuracy


def hypertune(n_trails, train_data, valid_data):

    objective_p = partial(objective, train_df=train_data, eval_df=valid_data)

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),  # seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective_p, n_trials=n_trails)  # , timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    print(f"====================================================")

    top3 = sorted(study.trials, key=lambda t: t.values[0], reverse=True)[:3]

    for t in top3:
        print(f"\tnumber: {t.number}")
        print(f"\tparams: {t.params}")
        print(f"\tuser_attr: {t.user_attrs}")
        print(f"\tvalues: {t.values}")
        print(f"====================================================")

    return [t.params for t in top3]


def train_lbgm(models_params, X_train, y_train, X_eval, y_eval):
    trained_models = []
    for p in models_params:
        clf = lgb.LGBMClassifier(**p, device="gpu")
        # X_train_T, _, _, y_train_T, _, _ = data_split(base_data, size=[5, 0, 3])
        # lgb_train_T = lgb.Dataset(X_train_T, y_train_T, free_raw_data=False, categorical_feature= [7, 14, 17, 24, 26])
        clf = clf.fit(X_train, y_train)
        # optuna does not save threshold
        clf_y_scores = clf.predict_proba(X_eval)[:, 1]
        thrs_value = select_threshold_5perc_fpr(y_eval, clf_y_scores)

        trained_models.append((clf, thrs_value))

    return trained_models


def calibrate(models_to_calibrate, X_eval, y_eval, method="isotonic"):

    calibrated_models = []

    for (clf, _) in models_to_calibrate:
        # decouple model original keeping it for further use.
        clf_ = deepcopy(clf)
        calibrated = CalibratedClassifierCV(clf_, cv="prefit", method=method).fit(
            X_eval, y_eval
        )
        calibrated_y_scores = calibrated.predict_proba(X_eval)[:, 1]
        calibrated_opt_thrshd = select_threshold_5perc_fpr(y_eval, calibrated_y_scores)

        calibrated_models.append((calibrated, calibrated_opt_thrshd))

    return calibrated_models
