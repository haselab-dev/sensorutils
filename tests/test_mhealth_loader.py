import sys
import unittest
import copy

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.mhealth import MHEALTH, load, load_raw, reformat


class MHEALTH_Test(unittest.TestCase):
    path = None
    activities = list(range(13))
    subjects = list(range(1, 10+1))

    data_columns = [
        'acceleration_chest_x', 'acceleration_chest_y', 'acceleration_chest_z',
        'electrocardiogram_1', 'electrocardiogram_2',
        'acceleration_left-ankle_x', 'acceleration_left-ankle_y', 'acceleration_left-ankle_z',
        'gyro_left-ankle_x', 'gyro_left-ankle_y', 'gyro_left-ankle_z',
        'magnetometer_left-ankle_x', 'magnetometer_left-ankle_y', 'magnetometer_left-ankle_z',
        'acceleration_right-lower-arm_x', 'acceleration_right-lower-arm_y', 'acceleration_right-lower-arm_z',
        'gyro_right-lower-arm_x', 'gyro_right-lower-arm_y', 'gyro_right-lower-arm_z',
        'magnetometer_right-lower-arm_x', 'magnetometer_right-lower-arm_y', 'magnetometer_right-lower-arm_z',
    ]
    meta_columns = ['activity', 'subject']

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = MHEALTH(self.path)

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
        self.assertTrue(all([list(m.columns) == MHEALTH_Test.meta_columns for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['activity'] == np.dtype(np.int8) and \
            m.dtypes['subject'] == np.dtype(np.int8)
            for m in meta
        ]))

        ## data check
        for col in MHEALTH_Test.meta_columns:
            self.assertTrue(all([
                set(np.unique(m[col])) == set([m[col].iloc[0]]) for m in meta
            ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([
            list(d.columns) == MHEALTH_Test.data_columns
            for d in data
        ]))

        ## data type check
        self.assertTrue(all([
            d.dtypes[c] == np.dtype(np.float64) \
            for d in data for c in data[0].columns
        ]))

    def test_load_raw_fn(self):
        raw = load_raw(self.path)

        # raw
        ## type check
        self.assertIsInstance(raw, list)
        self.assertTrue(all(isinstance(r, pd.DataFrame) for r in raw))

        ## shape and column check
        tgt = tuple(MHEALTH_Test.data_columns+MHEALTH_Test.meta_columns)
        self.assertTrue(all([
            tuple(r.columns) == tgt for r in raw
        ]))

        ## data type check
        flgs = []
        for r in raw:
            for col in r.columns:
                if col in ['activity', 'subject']:
                    flgs += [r.dtypes[col] == np.dtype(np.int8)]
                else:
                    flgs += [r.dtypes[col] == np.dtype(np.float64)]
        self.assertTrue(all(flgs))

        ## data check
        # for col in MHEALTH_Test.meta_columns:
        for col in ['subject']:
            self.assertTrue(all([
                set(np.unique(r[col])) == set([r[col].iloc[0]]) for r in raw
            ]))
        self.assertSetEqual(
            set(itertools.chain(*[np.unique(r['activity']).tolist() for r in raw])),
            set(self.activities)
        )
        self.assertSetEqual(
            set(itertools.chain(*[np.unique(r['subject']).tolist() for r in raw])),
            set(self.subjects)
        )

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
        self.assertTrue(all([list(m.columns) == MHEALTH_Test.meta_columns for m in meta]))

        ## data type check
        flgs = [m.dtypes[col] == np.dtype(np.int8) for m in meta for col in m.columns]
        self.assertTrue(all(flgs))

        ## data check
        for col in MHEALTH_Test.meta_columns:
            self.assertTrue(all([
                set(np.unique(m[col])) == set([m[col].iloc[0]]) for m in meta
            ]))
        self.assertSetEqual(
            set(itertools.chain(*[np.unique(m['activity']).tolist() for m in meta])),
            set(self.activities)
        )
        self.assertSetEqual(
            set(itertools.chain(*[np.unique(m['subject']).tolist() for m in meta])),
            set(self.subjects)
        )

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([
            list(d.columns) == \
            MHEALTH_Test.data_columns
            for d in data
        ]))

        ## data type check
        self.assertTrue(all([
            d.dtypes[c] == np.dtype(np.float64) \
            for d in data for c in data[0].columns
        ]))

    def test_mhealth_load_method_filed_labels(self):
        # check processing for exceptions
            
        x_labels = copy.deepcopy(MHEALTH_Test.data_columns)
        y_labels = copy.deepcopy(MHEALTH_Test.meta_columns)

        wsize = 64
        trim = 0
        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=wsize, stride=wsize, ftrim_sec=trim, btrim_sec=trim, subjects=None, x_labels=y_labels)

        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=wsize, stride=wsize, ftrim_sec=trim, btrim_sec=trim, subjects=None, y_labels=x_labels)

        # shape check
        ## x
        for n in range(10, len(x_labels), 10):
            x, _ = self.loader.load(window_size=wsize, stride=wsize, ftrim_sec=trim, btrim_sec=trim, subjects=None, x_labels=np.random.choice(x_labels, n, replace=False).tolist())
            self.assertTupleEqual(x.shape[1:], (n, wsize))
        ## y
        _, y = self.loader.load(window_size=wsize, stride=wsize, ftrim_sec=trim, btrim_sec=trim, subjects=None, y_labels=['activity'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=wsize, stride=wsize, ftrim_sec=trim, btrim_sec=trim, subjects=None, y_labels=['subject'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=wsize, stride=wsize, ftrim_sec=trim, btrim_sec=trim, subjects=None, y_labels=['activity', 'subject'])
        self.assertEqual(y.shape[1], 2)

    def test_mhealth_load_method_framing(self):
        x_labels = copy.deepcopy(MHEALTH_Test.data_columns)
        y_labels = copy.deepcopy(MHEALTH_Test.meta_columns)

        for stride, ws in itertools.product([64, 128, 256, 512], [64, 128, 256, 512]):
            # print(f'window size: {ws}, stride: {stride}')
            with self.subTest(f'window size: {ws}, stride: {stride}'):
                x, y = self.loader.load(window_size=ws, stride=stride, x_labels=None, y_labels=y_labels, ftrim_sec=0, btrim_sec=0, subjects=None)

                ## compare between x and y
                self.assertEqual(len(x), len(y))

                ## type check
                self.assertIsInstance(x, np.ndarray)
                self.assertIsInstance(y, np.ndarray)

                ## data type check
                self.assertEqual(x.dtype, np.dtype(np.float64))
                self.assertEqual(y.dtype, np.dtype(np.int8))

                ## shape check
                ## x_labels, y_labelsによって変わるため要検討
                self.assertEqual(len(x.shape), 3)
                self.assertTupleEqual(x.shape[1:], (len(x_labels), ws))
                self.assertEqual(len(y.shape), 2)
                self.assertEqual(y.shape[1], len(y_labels))

                ## data check
                ### x_labels, y_labelsによって変わるため要検討
                ### otherのデータが除去されていることも確認
                self.assertSetEqual(set(np.unique(y[:, 0])), set(self.activities)-set([0])) # remove null class
                self.assertSetEqual(set(np.unique(y[:, 1])), set(self.subjects)) # subject

                del x, y

    def test_mhealth_load_method_filed_subjects(self):
        patterns = [
            [10, 2, 6, 4, 8, 1, 9, 7, 5],
            [3, 10, 9, 2, 8, 7, 4, 1, 6, 5],
            [9, 7, 6, 10],
            [10, 8, 7, 2, 4, 5],
            [5, 9],
            [9, 6, 7, 2, 4, 5, 3, 10, 8],
            [9, 7, 3, 2, 5],
            [1, 2, 5, 3, 7, 8, 6],
            [5],
            [2, 3, 4],
        ]

        for i, subjects in enumerate(patterns):
            with self.subTest(f'pattern {i}'):
                _, y = self.loader.load(window_size=64, stride=64, ftrim_sec=0, btrim_sec=0, subjects=subjects)
                self.assertSetEqual(set(np.unique(y[:, 1])), set(subjects))


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    MHEALTH_Test.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
