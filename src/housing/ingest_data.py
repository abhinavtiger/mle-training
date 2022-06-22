"""
This module contains helper functions for ingestion of data.
Running this standalone downloads the housing data and stores preprocessed copies of it in the specified folders.
"""

import argparse
import os
import tarfile
from logging import Logger

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from housing.logger import configure_logger

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
imputer = SimpleImputer(strategy="median")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Function to download and extract housing data.
    Parameters
    ----------
    housing_url : str
        Url to download the housing data from.
    housing_path : str
        Path to store the raw csv files after extraction.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def train_test(housing):

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    """Does stratified shuffle split on "income_cat" attribute of housing data.
    Parameters
    ----------
    base_df : pd.DataFrame
        The dataframe to be split.
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        [train_dataset, test_dataset]
    """

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def preprocess(strat_train_set):
    housing = strat_train_set.copy()
    """Preprocesses the given dataframe. Imputes missing values with median.
    Replaces categorical column "ocean_proximity" with onehot dummy variables.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to preprocess.
   
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Index 0 is the preprocessed dataframe.
        Index 1 is the processed series
    """

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    return housing_prepared, housing_labels


def parse_args():
    """Commandline argument parser for standalone run.
    Returns
    -------
    arparse.Namespace
        Commandline arguments. Contains keys: ["raw": str,
         "processed": str,
         "log_level": str,
         "no_console_log": bool,
         "log_path": str]
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="path to store the dataset ",
        type=str,
        default="data/raw/housing",
    )
    parser.add_argument(
        "--dataprocessed",
        help="path to store the dataset ",
        type=str,
        default="data/processed",
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default=get_path() + "logs/logs.log")
    return parser.parse_args()


def get_path():
    # gets the current working directory
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "mle-training":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def save_preprocessed(train_X, train_y, test_X, test_y, processed):
    """To save the data in train and test
    data in data folder proecssed file"""
    train_X.to_csv(os.path.join(processed, "train_X.csv"), index=False)
    train_y.to_csv(os.path.join(processed, "train_y.csv"), index=False)
    test_X.to_csv(os.path.join(processed, "test_X.csv"), index=False)
    test_y.to_csv(os.path.join(processed, "test_y.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    logger = configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )
    parent_path = get_path()
    path = parent_path + args.datapath
    fetch_housing_data(housing_path=path)
    logger.debug("Fetched housing data.")
    logger.debug(f"Dataset stored at {path}.")
    data = load_housing_data(housing_path=path)
    logger.debug("Loaded housing data.")
    train, test = train_test(data)
    train_X, train_y = preprocess(train)
    print(train_X.shape, train_y.shape)
    logger.debug("Preprocessing housing data...")
    test_X, test_y = preprocess(test)
    processed = parent_path + args.dataprocessed
    if not os.path.exists(processed):
        os.makedirs(processed)
    save_preprocessed(train_X, train_y, test_X, test_y, processed)
    logger.debug(f"Preprocessed train and test datasets stored at {processed}.")
