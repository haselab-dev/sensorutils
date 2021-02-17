import unittest

import numpy as np
import pandas as pd
from pathlib import Path

import sys
# sys.path.append('../..') # cd sensorutils/tests; python HHAR_test.py
sys.path.append('..') # cd sensorutils; python tests/HHAR_test.py
import sensorutils.datasets.HHAR as HHAR


class HHARTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # 初期化処理 - 1度のみ
        ds_path = Path(r"C:\Users\daisuke\Documents\テーマ_メタ学習\code\data\hhar\Activity recognition exp")
        cls.segments = HHAR.load(ds_path)

        # print
        print(len(cls.segments)) # 477 != 486?
        print(cls.segments[0].head())

    def setUp(self):
        # 初期化処理 - unittestごと
        # ds_path = Path(r"C:\Users\daisuke\Documents\テーマ_メタ学習\code\data\hhar\Activity recognition exp")
        # ds_path = Path('../data/hhar/Activity recognition exp')

        # self.hhar = HHAR.HHAR(ds_path)
        # self.hhar.load()
        pass

    def tearDown(self):
        # 終了処理
        pass
        
    def test_HHAR_segments_same_User_in_seg(self):
        segments = self.segments.copy()

        flg = True
        for seg in segments:
            seg_labels = seg['User'].values
            flg *= np.all(seg_labels == seg_labels[0])
        self.assertTrue(flg)
        
    def test_HHAR_segments_same_Model_in_seg(self):
        segments = self.segments.copy()

        flg = True
        for seg in segments:
            seg_labels = seg['Model'].values
            flg *= np.all(seg_labels == seg_labels[0])
        self.assertTrue(flg)
        
    def test_HHAR_segments_same_Device_in_seg(self):
        segments = self.segments.copy()

        # flist = []
        # for i, seg in enumerate(segments):
        #     seg_labels = seg['Device'].values
        #     f = np.all(seg_labels == seg_labels[0])
        #     if not f: print(i, np.all(seg_labels == seg_labels[0]))
        #     flist.append(np.all(seg_labels == seg_labels[0]))
        # test = [x for i, x in enumerate(segments) if not flist[i]]
        # print(test)


        flg = True
        for seg in segments:
            seg_labels = seg['Device'].values
            flg *= np.all(seg_labels == seg_labels[0])
        self.assertTrue(flg)
        
    def test_HHAR_segments_same_act_in_seg(self):
        segments = self.segments.copy()

        flg = True
        for seg in segments:
            seg_labels = seg['gt'].values
            flg *= np.all(seg_labels == seg_labels[0])
        self.assertTrue(flg)

    def test_HHAR_segments_same_index_not_in_seg(self):
        segments = self.segments.copy()

        flg = True
        for seg in segments:
            hist = seg['Index'].value_counts()
            flg *= np.all(hist == 1)
        self.assertTrue(flg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
    # unittest.main(defaultTest='HHARTest.test_HHAR_segments_same_Model_in_seg', verbosity=2)
    # unittest.main(defaultTest='HHARTest.test_HHAR_segments_same_Device_in_seg', verbosity=2)