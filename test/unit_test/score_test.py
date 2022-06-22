import os
from turtle import mode

import housing.score as score

import unittest
import os

args = score.parse_args()
rootpath = score.get_path()

class TestTrain(unittest.TestCase):
    def test_parse_args(self):

        self.assertTrue( args.datapath == "data/processed")
        self.assertTrue (args.modelpath == "artifacts")
        self.assertTrue (args.log_level == "DEBUG")
        self.assertFalse(args.no_console_log)
        self.assertTrue (args.log_path == rootpath+"logs/logs.log")

    def test_load_data(self):
        test_X,test_y=score.load_data(rootpath+args.datapath)
        self.assertTrue(len(test_X) == len(test_y))
        self.assertTrue(len(test_y.shape) == 1)

    def test_load_models(self):
        models = score.load_models(rootpath+args.modelpath)
        self.assertTrue(len(models) == 4)



if __name__ == '__main__':
    unittest.main()