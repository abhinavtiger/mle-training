import unittest
from housing import ingest_data as data
import os
import pandas as pd


args = data.parse_args()
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = args.datapath
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
rootpath = data.get_path()

class Testutils(unittest.TestCase):
    def test_parse_args(self):

        self.assertTrue( args.datapath == "data/raw/housing")
        self.assertTrue (args.dataprocessed == "data/processed")
        self.assertTrue (args.log_level == "DEBUG")
        self.assertFalse(args.no_console_log)
        self.assertTrue (args.log_path == rootpath+"logs/logs.log")


    def test_fetch_data(self):
        data.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
        self.assertTrue(os.path.isfile(f"{rootpath}{args.datapath}/housing.tgz"))
        self.assertTrue (os.path.isfile(f"{rootpath}{args.datapath}/housing.csv"))


    def test_split(self):
        housing_df = pd.read_csv(f"{rootpath}{args.datapath}/housing.csv")
        train_set, test_set = data.train_test(housing_df)
        self.assertFalse (train_set.isna().sum().sum() == 0)
        self.assertFalse (test_set.isna().sum().sum() == 0)
        self.assertTrue (len(train_set) == len(housing_df) * 0.8)
        self.assertTrue (len(test_set) == len(housing_df) * 0.2)


    def test_preprocess(self):
        housing_df = pd.read_csv(f"{rootpath}{args.datapath}/housing.csv")
        train_set, test_set = data.train_test(housing_df)
        train_X,train_y = data.preprocess(train_set)
        test_X, test_y= data.preprocess(test_set)
        
        self.assertTrue ("ocean_proximity" not in train_X.columns)
        self.assertTrue ("ocean_proximity" not in test_X.columns)
        self.assertTrue ("rooms_per_household" in train_X.columns)
        self.assertTrue ("rooms_per_household" in test_X.columns)
        self.assertTrue ("population_per_household" in train_X.columns)
        self.assertTrue ("population_per_household" in test_X.columns)
        self.assertTrue ("bedrooms_per_room" in train_X.columns)
        self.assertTrue ("bedrooms_per_room" in test_X.columns)

if __name__ == '__main__':
    unittest.main()