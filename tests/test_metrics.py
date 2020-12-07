import unittest
import numpy as np
from metrics import (
    mae,
    mse,
    mape,
    rmse,
    rmspe,
    rmsle,
    r2,
    snr,
    lsd,
)

class TestMSE(unittest.TestCase):
    def test_1(self):
        a = np.arange(10)
        b = np.arange(10)
        res1 = mse(a, b)
        self.assertEqual(res1, 0)
        res2 = mse(b, a)
        self.assertEqual(res1, res2)
        return

    def test_2(self):
        a = np.arange(5)
        b = np.array([5] * 5)
        res = mse(a, b)
        self.assertEqual(res, 11)
        return

class TestMAE(unittest.TestCase):
    def test_1(self):
        a = np.arange(10)
        b = np.arange(10)
        res1 = mae(a, b)
        self.assertEqual(res1, 0)
        res2 = mae(b, a)
        self.assertEqual(res1, res2)
        return

    def test_2(self):
        a = np.arange(5)
        b = np.array([5] * 5)
        res = mae(a, b)
        self.assertEqual(res, 3)
        return

if __name__ == '__main__':
    unittest.main()
