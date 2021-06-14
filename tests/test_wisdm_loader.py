import sys
import unittest

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.wisdm import WISDM, load, load_raw, reformat


class WISDMTest(unittest.TestCase):
    path = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = WISDM(self.path)

    def tearDown(self):
        pass

    def test_load_fn(self):
        data, meta = load(self.path)

        # compare between data and meta
        self.assertEqual(len(data), len(meta))

        # check meta
        ## type check
        self.assertIsInstance(meta, list)
        self.assertTrue(all(isinstance(m, pd.DataFrame) for m in meta))

        ## shape and column check
        self.assertTrue(all([set(m.columns) == set(['user', 'activity', 'timestamp']) for m in meta]))

        ## data type check
        self.assertTrue(all(list(
            itertools.chain(*[
                [dt == np.dtype(np.uint8) for dt in m.dtypes[:-1]] for m in meta
            ])
        )))
        self.assertTrue(all([m.dtypes[-1] == np.dtype(np.uint64) for m in meta]))

        ## data check
        self.assertTrue(all([
            [set(np.unique(m['user'])) == set([m['user'].iloc[0]]) for m in meta]
        ]))
        self.assertTrue(all([
            [set(np.unique(m['activity'])) == set([m['activity'].iloc[0]]) for m in meta]
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([len(d.shape) == 2 for d in data]))
        self.assertTrue(all([d.shape[1] == 3 for d in data]))
        self.assertTrue(all([set(d.columns) == set(['x-acceleration', 'y-acceleration', 'z-acceleration']) for d in data]))

    def test_load_raw_fn(self):
        raw = load_raw(self.path)

        # raw
        ## type check
        self.assertIsInstance(raw, pd.DataFrame)

        ## shape and column check
        self.assertSetEqual(set(raw.columns), set(['user', 'activity', 'timestamp', 'x-acceleration', 'y-acceleration', 'z-acceleration']))

        ## data type check
        self.assertTupleEqual(tuple(raw.dtypes), tuple(['uint8', 'uint8', 'uint64', 'float64', 'float64', 'float64']))

        ## data check
        self.assertSetEqual(set(np.unique(raw['user'])), set(range(1, 36+1)))
        self.assertSetEqual(set(np.unique(raw['activity'])), set(range(6)))

    def test_reformat_fn(self):
        raw = load_raw(self.path)
        data, meta = reformat(raw)

        # compare between data and meta
        self.assertEqual(len(data), len(meta))

        # check meta
        ## type check
        self.assertIsInstance(meta, list)
        self.assertTrue(all(isinstance(m, pd.DataFrame) for m in meta))

        ## shape and column check
        self.assertTrue(all([set(m.columns) == set(['user', 'activity', 'timestamp']) for m in meta]))

        ## data type check
        self.assertTrue(all(list(
            itertools.chain(*[
                [dt == np.dtype(np.uint8) for dt in m.dtypes[:-1]] for m in meta
            ])
        )))
        self.assertTrue(all([m.dtypes[-1] == np.dtype(np.uint64) for m in meta]))

        ## data check
        self.assertTrue(all([
            [set(np.unique(m['user'])) == set([m['user'].iloc[0]]) for m in meta]
        ]))
        self.assertTrue(all([
            [set(np.unique(m['activity'])) == set([m['activity'].iloc[0]]) for m in meta]
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([len(d.shape) == 2 for d in data]))
        self.assertTrue(all([d.shape[1] == 3 for d in data]))
        self.assertTrue(all([set(d.columns) == set(['x-acceleration', 'y-acceleration', 'z-acceleration']) for d in data]))

    def test_wisdm_load_method_framing(self):
        for stride, ws in itertools.product([64, 128, 256, 512], [64, 128, 256, 512]):
            with self.subTest(f'window size: {ws}'):
                x, y = self.loader.load(window_size=ws, stride=stride, ftrim_sec=2, btrim_sec=2, subjects=None)

                ## compare between x and y
                self.assertEqual(len(x), len(y))

                ## type check
                self.assertIsInstance(x, np.ndarray)
                self.assertIsInstance(y, np.ndarray)

                ## data type check
                self.assertEqual(x.dtype, np.dtype(np.float64))
                self.assertEqual(y.dtype, np.dtype(np.uint8))

                ## shape check
                self.assertEqual(len(x.shape), 3)
                self.assertTupleEqual(x.shape[1:], (3, ws))
                self.assertEqual(len(y.shape), 2)
                self.assertEqual(y.shape[1], 2)

                ## data check
                self.assertSetEqual(set(np.unique(y[:, 0])), set(range(6))) # activity
                self.assertSetEqual(set(np.unique(y[:, 1])), set(range(1, 36+1))) # subject

    def test_wisdm_load_method_filed_subjects(self):
        patterns = [
            [23],
            [23, 20, 17, 25, 31, 29, 28, 15, 8, 13, 36, 27, 14, 34, 35, 22, 16, 18, 9, 7, 6, 5, 26, 1, 3, 30, 19, 12, 4, 24, 21, 11, 32],
            [26],
            [6, 36, 14, 17, 13, 31, 8, 34, 20, 25, 9, 7, 16, 5],
            [23, 28, 16, 8, 2, 19, 6, 11, 18, 24, 10, 7, 35, 30],
            [18, 21, 5, 1, 15, 6, 30, 26, 11, 20, 17, 3, 4, 28, 31, 24, 13, 2, 35, 36],
            [18, 14, 16, 9, 1, 7, 35, 5, 2, 6, 17],
            [1],
            [3, 2, 13, 7, 25, 36, 6],
            [35, 16, 10, 3, 27, 7, 28, 22, 2, 13, 25, 30, 33, 17, 24, 32, 6, 36, 9, 8, 20, 4, 31, 19],
        ]

        for i, subjects in enumerate(patterns):
            with self.subTest(f'pattern {i}'):
                _, y = self.loader.load(window_size=256, stride=256, ftrim_sec=2, btrim_sec=2, subjects=subjects)
                self.assertSetEqual(set(np.unique(y[:, 1])), set(subjects))


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    WISDMTest.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
