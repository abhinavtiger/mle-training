import os
from turtle import mode

import housing.train as train

import unittest
import os

args = train.parse_args()
rootpath = train.get_path()

class TestTrain(unittest.TestCase):
    def test_parse_args(self):

        self.assertTrue( args.inputpath == "data/processed/")
        self.assertTrue (args.outputpath == "artifacts")
        self.assertTrue (args.log_level == "DEBUG")
        self.assertFalse(args.no_console_log)
        self.assertTrue (args.log_path == rootpath+"logs/logs.log")

    def test_load_data(self):
        train_X,train_y=train.load_data(rootpath+args.inputpath)
        self.assertTrue(len(train_X) == len(train_y))
        self.assertTrue(len(train_y.shape) == 1)

    def test_save_model(self):
        models=train.model_names
        for i in models:
            self.assertTrue(os.path.isfile(f"{rootpath}{args.outputpath}/models/{i}.pkl"))



if __name__ == '__main__':
    unittest.main()