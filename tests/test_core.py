import unittest
from unittest.case import skip

import numpy as np

import sys
sys.path.append('.')
from src.sensorutils import core


class TestCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # 初期化処理 - 1度のみ
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        # 終了処理 - 1度のみ
        pass

    def setUp(self):
        # 初期化処理 - unittestごと
        pass

    def tearDown(self):
        # 終了処理 - unittestごと
        pass


    @skip("Not Implemented")
    def test_to_frames(self):
        flg = True
        self.assertTrue(flg)

    @skip("Not Implemented")
    def test_to_frames_using_reshape(self):
        flg = True
        self.assertTrue(flg)

    @skip("Not Implemented")
    def test_to_frames_using_index(self):
        flg = True
        self.assertTrue(flg)

    @skip("Not Implemented")
    def test_to_frames_using_nptricks(self):
        flg = True
        self.assertTrue(flg)

    @skip("Not Implemented")
    def test_split_using_sliding_window(self):
        flg = True
        self.assertTrue(flg)

    def test_split_using_target(self):

        tgt = np.array([0, 0, 1, 1, 2, 2, 1])
        src = np.array([1, 2, 3, 4, 5, 6, 7])
        ans = {0: [np.array([1, 2])], 1: [np.array([3, 4]), np.array([7])], 2: [np.array([5, 6])]}

        result = core.split_using_target(src, tgt)

        flg = True
        flg &= (ans.keys() == result.keys())
        # print(ans.keys(), result.keys())

        # 辞書型の要素の値を比較
        for a_v, r_v in zip(ans.values(), result.values()):
            # print(a_v, r_v)
            for a_array, r_array in zip(a_v, r_v):
                # print("\t", a_array, r_array)
                flg &= np.all(a_array == r_array)

        self.assertTrue(flg)
        
    @skip("Not Implemented")
    def test_interpolate(self):
        flg = True
        self.assertTrue(flg)
        
    @skip("Not Implemented")
    def test_pickle_dump(self):
        flg = True
        self.assertTrue(flg)
        
    @skip("Not Implemented")
    def test_pickle_load(self):
        flg = True
        self.assertTrue(flg)

if __name__ == "__main__":
    unittest.main(verbosity=2)