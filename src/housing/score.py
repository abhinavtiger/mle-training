"""
This module contains helper functions to score the models.
Can be run standalone with commandline arguments for models and the datasets to score them on.
"""
import argparse
import os
import pickle
from logging import Logger

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from housing.logger import configure_logger

model_names = ["lin_model", "tree_model", "forest_model", "grid_search_model"]


def get_path():
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle-training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def parse_args():
    """Commandline argument parser for standalone run.
    Returns
    -------
    arparse.Namespace
        Commandline arguments. Contains keys: [
         "dataset": str,
         "models": str,
         "log_level": str,
         "no_console_log": bool,
         "log_path": str]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath", help="path to the datasets ", type=str, default="data/processed"
    )
    parser.add_argument(
        "--modelpath", help="path to the model files ", type=str, default="artifacts"
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default=get_path() + "logs/logs.log")
    return parser.parse_args()


def scoring(X_test, y_test, lin_reg, tree_reg, forest_reg, grid_search):
    """Predict the models based on scores
    scores:RMSE,MAE"""

    lin_predictions = lin_reg.predict(X_test)
    lin_mse = mean_squared_error(y_test, lin_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y_test, lin_predictions)

    tree_predictions = tree_reg.predict(X_test)
    tree_mse = mean_squared_error(y_test, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(y_test, tree_predictions)

    forest_predictions = forest_reg.predict(X_test)
    forest_mse = mean_squared_error(y_test, forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_mae = mean_absolute_error(y_test, forest_predictions)

    grid_search_predictions = grid_search.predict(X_test)
    grid_search_mse = mean_squared_error(y_test, grid_search_predictions)
    grid_search_rmse = np.sqrt(grid_search_mse)
    grid_search_mae = mean_absolute_error(y_test, grid_search_predictions)

    lin_scores = [lin_mae, lin_mse, lin_rmse]
    tree_scores = [tree_mae, tree_mse, tree_rmse]
    forest_scores = [forest_mae, forest_mse, forest_rmse]
    grid_search_scores = [grid_search_mae, grid_search_mse, grid_search_rmse]

    return lin_scores, tree_scores, forest_scores, grid_search_scores


def load_data(in_path):
    prepared = pd.read_csv(in_path + "/test_X.csv")
    lables = pd.read_csv(in_path + "/test_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


def load_models(model_path):
    """Loads models from given directory path.
    Parameters
    ----------
    path : str
        Path to directory with model pkl files.
    Returns
    -------
    list[sklearn.base.BaseEstimator]
        List of models loaded from pkl files in directory.
    """
    models = []
    for i in model_names:
        with open(model_path + "/models/" + i + ".pkl", "rb") as f:
            models.append(pickle.load(f))
    return models


def score(models, X_test, y_test):

    """Scores given model on given data.
     Parameters
     ----------
     model : sklearn.base.BaseEstimator
         Estimator to score.
     X : pd.DataFrame
         Input features dataframe.
     y : pd.Series
         Ground truth labels.

     Returns
     -------
    list[sklearn.base.BaseEstimator]
         List of models loaded from pkl files in directory.
    """

    lin_scores, tree_scores, forest_scores, grid_search_scores = scoring(
        X_test, y_test, models[0], models[1], models[2], models[3]
    )

    return [lin_scores, tree_scores, forest_scores, grid_search_scores]


if __name__ == "__main__":
    """Runs the whole scoring process"""

    args = parse_args()
    logger = configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )
    path_parent = get_path()
    data_path = path_parent + args.datapath
    model_path = path_parent + args.modelpath
    X_test, y_test = load_data(data_path)
    logger.debug("Loaded test data")
    models = load_models(model_path)
    logger.debug("Loaded Models")
    scores = []
    scores = score(models, X_test, y_test)
    for i in range(len(models)):
        logger.debug(f"{model_names[i]}={scores[i]}")
