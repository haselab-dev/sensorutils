import sys
import unittest

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.usc_had import USC_HAD, load, load_raw, reformat


class USC_HAD_Test(unittest.TestCase):
    path = None
    activities = list(range(12))
    subjects = list(range(1, 14+1))

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = USC_HAD(self.path)

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
        self.assertTrue(all([list(m.columns) == ['version', 'trial', 'activity', 'subject', 'age', 'height', 'weight', 'sensor_location', 'sensor_orientation'] for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['version'] == np.dtype(np.int8) and \
            m.dtypes['trial'] == np.dtype(np.int8) and \
            m.dtypes['activity'] == np.dtype(np.int8) and \
            m.dtypes['subject'] == np.dtype(np.int8) and \
            m.dtypes['age'] == np.dtype(np.int8) and \
            m.dtypes['height'] == np.dtype(np.float64) and \
            m.dtypes['weight'] == np.dtype(np.float64) and \
            m.dtypes['sensor_location'] == np.dtype(np.dtype(object)) and \
            m.dtypes['sensor_orientation'] == np.dtype(np.dtype(object))
            for m in meta
        ]))

        ## data check
        for col in ['version', 'trial', 'activity', 'subject', 'age', 'height', 'weight', 'sensor_location', 'sensor_orientation']:
            self.assertTrue(all([
                set(np.unique(m[col])) == set([m[col].iloc[0]]) for m in meta
            ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([
            list(d.columns) == \
            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
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
        tgt = tuple([
            'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'version', 'trial',
            'activity', 'subject',
            'age', 'height', 'weight',
            'sensor_location', 'sensor_orientation',
        ])
        self.assertTrue(all([
            tuple(r.columns) == tgt for r in raw
        ]))

        ## data type check
        flgs = []
        for r in raw:
            for col in r.columns:
                if col in ['version', 'trial', 'activity', 'subject', 'age']:
                    flgs += [r.dtypes[col] == np.dtype(np.int8)]
                elif col in ['sensor_location', 'sensor_orientation']:
                    flgs += [r.dtypes[col] == np.dtype(object)]
                else:
                    flgs += [r.dtypes[col] == np.dtype(np.float64)]
        self.assertTrue(all(flgs))

        ## data check
        for col in ['version', 'trial', 'activity', 'subject', 'age', 'height', 'weight', 'sensor_location', 'sensor_orientation']:
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
        self.assertTrue(all([list(m.columns) == ['version', 'trial', 'activity', 'subject', 'age', 'height', 'weight', 'sensor_location', 'sensor_orientation'] for m in meta]))

        ## data type check
        flgs = []
        for m in meta:
            for col in m.columns:
                if col in ['version', 'trial', 'activity', 'subject', 'age']:
                    flgs += [m.dtypes[col] == np.dtype(np.int8)]
                elif col in ['sensor_location', 'sensor_orientation']:
                    flgs += [m.dtypes[col] == np.dtype(object)]
                else:
                    flgs += [m.dtypes[col] == np.dtype(np.float64)]
        self.assertTrue(all(flgs))

        ## data check
        for col in ['version', 'trial', 'activity', 'subject', 'age', 'height', 'weight', 'sensor_location', 'sensor_orientation']:
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
            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            for d in data
        ]))

        ## data type check
        self.assertTrue(all([
            d.dtypes[c] == np.dtype(np.float64) \
            for d in data for c in data[0].columns
        ]))

    def test_usc_had_load_method_filed_labels(self):
        # check processing for exceptions
            
        x_labels = [
            'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
        ]
        y_labels = [
            'version', 'trial',
            'activity', 'subject',
            'age', 'height', 'weight',
            'sensor_location', 'sensor_orientation',
        ]

        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=None, x_labels=y_labels)

        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=None, y_labels=x_labels)

        # shape check
        ## x
        for n in range(10, len(x_labels), 10):
            x, _ = self.loader.load(window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=None, x_labels=np.random.choice(x_labels, n, replace=False).tolist())
            self.assertTupleEqual(x.shape[1:], (n, 256))
        ## y
        _, y = self.loader.load(window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=None, y_labels=['activity'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=None, y_labels=['subject'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=None, y_labels=['activity', 'subject'])
        self.assertEqual(y.shape[1], 2)

    def test_usc_had_load_method_framing(self):
        x_labels = [
            'acc_x', 'acc_y', 'acc_z',
            'gyro_x', 'gyro_y', 'gyro_z',
        ]
        y_labels = [
            'version', 'trial',
            'activity', 'subject',
            'age', 'height', 'weight',
            'sensor_location', 'sensor_orientation',
        ]
        for stride, ws in itertools.product([64, 128, 256, 512], [64, 128, 256, 512]):
            # print(f'window size: {ws}, stride: {stride}')
            with self.subTest(f'window size: {ws}, stride: {stride}'):
                x, y = self.loader.load(window_size=ws, stride=stride, x_labels=None, y_labels=y_labels, ftrim_sec=2, btrim_sec=2, subjects=None)

                ## compare between x and y
                self.assertEqual(len(x), len(y))

                ## type check
                self.assertIsInstance(x, np.ndarray)
                self.assertIsInstance(y, np.ndarray)

                ## data type check
                self.assertEqual(x.dtype, np.dtype(np.float64))
                # self.assertEqual(y.dtype, np.dtype(np.int8))  # USC-HADではnp.int8とは限らない

                ## shape check
                ## x_labels, y_labelsによって変わるため要検討
                self.assertEqual(len(x.shape), 3)
                self.assertTupleEqual(x.shape[1:], (len(x_labels), ws))
                self.assertEqual(len(y.shape), 2)
                self.assertEqual(y.shape[1], len(y_labels))

                ## data check
                ### x_labels, y_labelsによって変わるため要検討
                ### otherのデータが除去されていることも確認
                self.assertSetEqual(set(np.unique(y[:, 2])), set(self.activities))
                self.assertSetEqual(set(np.unique(y[:, 3])), set(self.subjects)) # subject

                del x, y

    def test_pamap2_load_method_filed_subjects(self):
        patterns = [
            [4, 3, 1, 6, 5, 8, 14, 11, 12, 7, 10, 13],
            [3, 11, 14, 6, 12, 5],
            [13, 3, 8, 9, 12, 5, 10, 14],
            [7, 13],
            [11, 8, 7, 1, 2, 6, 5, 9, 14, 3],
            [6, 12, 8, 1],
            [8, 4, 11, 6, 7, 9, 5, 2],
            [3, 10, 12, 2, 13, 7, 11, 1, 6],
            [13, 1, 6, 5, 11],
            [2, 11, 5, 14],
        ]

        for i, subjects in enumerate(patterns):
            with self.subTest(f'pattern {i}'):
                _, y = self.loader.load(window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=subjects)
                self.assertSetEqual(set(np.unique(y[:, 3])), set(subjects))


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    USC_HAD_Test.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
