import sys
import unittest

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.unimib import UniMib, load, load_raw, reformat


class UniMibTest(unittest.TestCase):
    path = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = UniMib(self.path)

    def tearDown(self):
        pass

    def test_load_fn(self):
        def _check_common(self, data, meta, dtype):
            # compare between data and meta
            self.assertEqual(len(data), len(meta))

            # meta
            ## type check
            self.assertIsInstance(meta, pd.DataFrame)

            ## shape and column check
            if dtype in ['full', 'adl', 'fall']:
                self.assertSetEqual(set(meta.columns), set(['activity', 'subject', 'trial_id']))
            elif dtype == 'raw':
                self.assertSetEqual(set(meta.columns), set(['activity', 'subject', 'trial_id', 'gender', 'age', 'height', 'weight']))
            else:
                self.fail(f'Unexpected case, dtype: {dtype}')
            
            ## data type check
            # flags_dtype = [dt == np.dtype(np.int8) or dt == np.dtype(np.int16) or dt == np.dtype(np.int32) or dt == np.dtype(np.int64) for dt in meta.dtypes]
            flags_dtype = [dt == np.dtype(np.int8) for dt in meta.dtypes]
            self.assertTrue(all(flags_dtype))

            ## data check
            if dtype == 'full':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 17+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            elif dtype == 'adl':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 9+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 2+1)))
            elif dtype == 'fall':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 8+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            elif dtype == 'raw':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 17+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            else:
                self.fail(f'Unexpected case, dtype: {dtype}')

            # data
            ## type check
            self.assertIsInstance(data, list)
            self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

            ## shape check
            flgs_shape = [len(d.shape) == 2 for d in data]
            self.assertTrue(all(flgs_shape))
            flgs_shape_ax1 = [d.shape[1] == 3 for d in data]
            self.assertTrue(all(flgs_shape_ax1))
            self.assertTrue(all(
                [set(d.columns) == set(['x', 'y', 'z']) for d in data]
            ))

        data_types = ['full', 'adl', 'fall']
        for dtype in data_types:
            with self.subTest(f'data type: {dtype}'):
                data, meta = load(self.path, data_type=dtype)

                _check_common(self, data, meta, dtype)

                # data
                ## data check
                sizes_seg = set(len(d) for d in data)
                self.assertEqual(len(sizes_seg), 1)

        with self.subTest('data type: raw'):
            data, meta = load(self.path, data_type='raw')

            _check_common(self, data, meta, 'raw')

    def test_load_raw_fn(self):
        def _check_common(self, data, meta, dtype):
            # compare between data and meta
            self.assertEqual(len(data), len(meta))

            # meta
            ## type check
            self.assertIsInstance(meta, pd.DataFrame)
            if dtype in ['full', 'adl', 'fall']:
                ## meta - shape and column check
                self.assertSetEqual(set(meta.columns), set(['activity', 'subject', 'trial_id']))

                ## data - type check
                self.assertIsInstance(data, np.ndarray)
                self.assertTrue(all(isinstance(d, np.ndarray) for d in data))
            elif dtype == 'raw':
                ## meta - shape and column check
                self.assertSetEqual(set(meta.columns), set(['activity', 'subject', 'trial_id', 'gender', 'age', 'height', 'weight']))

                ## data - type check
                self.assertIsInstance(data, list)
                self.assertTrue(all(isinstance(d, np.ndarray) for d in data))
            else:
                self.fail(f'Unexpected case, dtype: {dtype}')

            ## data type check
            # flags_dtype = [dt == np.dtype(np.int8) or dt == np.dtype(np.int16) or dt == np.dtype(np.int32) or dt == np.dtype(np.int64) for dt in meta.dtypes]
            flags_dtype = [dt == np.dtype(np.int8) for dt in meta.dtypes]
            self.assertTrue(all(flags_dtype))

            ## data check
            if dtype == 'full':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 17+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            elif dtype == 'adl':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 9+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 2+1)))
            elif dtype == 'fall':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 8+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            elif dtype == 'raw':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 17+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            else:
                self.fail(f'Unexpected case, dtype: {dtype}')

            # data
            ## type check
            if dtype in ['full', 'adl', 'fall']:
                self.assertIsInstance(data, np.ndarray)
            elif dtype == 'raw':
                self.assertIsInstance(data, list)
                self.assertTrue(all(isinstance(d, np.ndarray) for d in data))
            else:
                self.fail(f'Unexpected case, dtype: {dtype}')

            ## shape check
            flgs_shape = [len(d.shape) == 2 for d in data]
            self.assertTrue(all(flgs_shape))
            flgs_shape_ax1 = [d.shape[0] == 3 for d in data]    # different point
            self.assertTrue(all(flgs_shape_ax1))

        data_types = ['full', 'adl', 'fall']
        for dtype in data_types:
            with self.subTest(f'data type: {dtype}'):
                raw = load_raw(self.path, data_type=dtype)

                ## raw - type check
                self.assertIsInstance(raw, tuple)
                self.assertEqual(len(raw), 2)
                data, meta = raw

                _check_common(self, data, meta, dtype)

                # data
                ## data check
                sizes_seg = set(len(d) for d in data)
                self.assertEqual(len(sizes_seg), 1)

        with self.subTest('data type: raw'):
            raw = load_raw(self.path, data_type='raw')

            ## raw - type check
            self.assertIsInstance(raw, tuple)
            self.assertEqual(len(raw), 2)
            data, meta = raw

            _check_common(self, data, meta, 'raw')
    
    def test_reformat_fn(self):
        def _check_common(self, data, meta, dtype):
            # compare between data and meta
            self.assertEqual(len(data), len(meta))

            # meta
            ## type check
            self.assertIsInstance(meta, pd.DataFrame)

            ## shape and column check
            if dtype in ['full', 'adl', 'fall']:
                self.assertSetEqual(set(meta.columns), set(['activity', 'subject', 'trial_id']))
            elif dtype == 'raw':
                self.assertSetEqual(set(meta.columns), set(['activity', 'subject', 'trial_id', 'gender', 'age', 'height', 'weight']))
            else:
                self.fail(f'Unexpected case, dtype: {dtype}')

            ## data type check
            # flags_dtype = [dt == np.dtype(np.int8) or dt == np.dtype(np.int16) or dt == np.dtype(np.int32) or dt == np.dtype(np.int64) for dt in meta.dtypes]
            flags_dtype = [dt == np.dtype(np.int8) for dt in meta.dtypes]
            self.assertTrue(all(flags_dtype))

            ## data check
            if dtype == 'full':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 17+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            elif dtype == 'adl':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 9+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 2+1)))
            elif dtype == 'fall':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 8+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            elif dtype == 'raw':
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 17+1)))
                self.assertSetEqual(set(np.unique(meta['subject'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['trial_id'])), set(range(1, 6+1)))
            else:
                self.fail(f'Unexpected case, dtype: {dtype}')
            
            # data
            ## type check
            self.assertIsInstance(data, list)
            self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

            ## shape check
            flgs_shape = [len(d.shape) == 2 for d in data]
            self.assertTrue(all(flgs_shape))
            flgs_shape_ax1 = [d.shape[1] == 3 for d in data]    # different point
            self.assertTrue(all(flgs_shape_ax1))
            self.assertTrue(all(
                [set(d.columns) == set(['x', 'y', 'z']) for d in data]
            ))

        data_types = ['full', 'adl', 'fall']
        for dtype in data_types:
            with self.subTest(f'data type: {dtype}'):
                raw = load_raw(self.path, data_type=dtype)
                data, meta = reformat(raw)

                _check_common(self, data, meta, dtype)

                ## data - data type check
                sizes_seg = set(len(d) for d in data)
                self.assertEqual(len(sizes_seg), 1)

        with self.subTest('data type: raw'):
            raw = load_raw(self.path, data_type='raw')
            data, meta = reformat(raw)
            _check_common(self, data, meta, 'raw')

    def test_unimib_load_method_framing(self):
        data_types = ['full', 'adl', 'fall']
        for dtype in data_types:
            with self.subTest(f'data type: {dtype}'):
                x, y = self.loader.load(data_type=dtype, subjects=None)

                ## compare between x and y
                self.assertEqual(len(x), len(y))

                ## type check
                self.assertIsInstance(x, np.ndarray)
                self.assertIsInstance(y, np.ndarray)

                ## data type check
                self.assertEqual(x.dtype, np.dtype(np.float64))
                self.assertEqual(y.dtype, np.dtype(np.int8))

                ## shape check
                self.assertEqual(len(x.shape), 3)
                self.assertTupleEqual(x.shape[1:], (3, 151))
                self.assertEqual(len(y.shape), 2)
                self.assertEqual(y.shape[1], 2)

                ## data check
                if dtype == 'full':
                    self.assertSetEqual(set(np.unique(y[:, 0])), set(range(1, 17+1))) # activity
                elif dtype == 'adl':
                    self.assertSetEqual(set(np.unique(y[:, 0])), set(range(1, 9+1))) # activity
                elif dtype == 'fall':
                    self.assertSetEqual(set(np.unique(y[:, 0])), set(range(1, 8+1))) # activity
                self.assertSetEqual(set(np.unique(y[:, 1])), set(range(1, 30+1))) # subject

        for stride, ws in itertools.product([64, 128, 256, 512], [64, 128, 256, 512]):
            with self.subTest(f'window size: {ws}, data type: raw'):
                x, y = self.loader.load(data_type='raw', window_size=ws, stride=stride, ftrim_sec=2, btrim_sec=2, subjects=None)

                ## compare between x and y
                self.assertEqual(len(x), len(y))

                ## type check
                self.assertIsInstance(x, np.ndarray)
                self.assertIsInstance(y, np.ndarray)

                ## data type check
                self.assertEqual(x.dtype, np.dtype(np.float64))
                self.assertEqual(y.dtype, np.dtype(np.int8))

                ## shape check
                self.assertEqual(len(x.shape), 3)
                self.assertTupleEqual(x.shape[1:], (3, ws))
                self.assertEqual(len(y.shape), 2)
                self.assertEqual(y.shape[1], 2)

                ## data check
                ### window_sizeが大きいとラベルば網羅されない可能性がある
                ### ftrim_sec(btrim_sec)が大きいとfallラベルが入らなくなる可能性がある
                if ws == 64 and stride == 64:
                    self.assertSetEqual(set(np.unique(y[:, 0])), set(range(1, 17+1))) # activity
                    self.assertSetEqual(set(np.unique(y[:, 1])), set(range(1, 30+1))) # subject

    def test_unimib_load_method_filed_subjects(self):
        data_types = ['full', 'adl', 'fall']
        
        patterns = [
            [3, 30, 14, 11, 28, 26, 23, 12, 18, 24, 6, 17, 9],
            [14, 7, 26, 9, 6, 21, 30, 13, 12, 3, 27, 22, 5, 11, 28, 17],
            [22, 27, 13, 29, 24, 28, 30],
            [11, 7, 30, 13, 8, 20, 27, 29, 23, 12, 5, 3, 2, 25, 19, 1, 26, 10, 6, 18, 17],
            [14, 13, 8, 16, 6, 26, 29, 18, 23, 9, 7, 17, 5, 2, 28, 30, 20, 21, 15, 27, 22, 4],
            [7, 15, 16, 8, 2, 10, 21, 6, 22, 13, 1, 24, 5, 20, 23, 11, 18, 19, 28, 17],
            [19, 25, 12, 4, 11, 24, 6, 16, 13, 18, 27, 3, 10, 28, 21, 8, 14, 7, 5, 1, 9, 2, 20, 29, 23],
            [15, 14, 18, 10, 23, 2, 8, 21, 3, 26, 20, 17, 22, 24, 5, 9, 27, 1, 12, 6, 29, 7, 13, 11, 28],
            [7, 9, 5, 21, 20, 16, 17, 1, 18, 3, 2, 6, 22, 28, 14, 12, 11, 25, 23, 13, 19, 15, 29, 24, 26, 27, 10, 8, 30, 4],
            [15, 7, 10, 1, 21, 19, 11, 29, 20, 12, 14, 16],
        ]
        for i, subjects in enumerate(patterns):
            for dtype in data_types:
                with self.subTest(f'pattern {i}, data type: {dtype}'):
                    _, y = self.loader.load(data_type=dtype, window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=subjects)
                    ## data check
                    self.assertSetEqual(set(np.unique(y[:, 1])), set(subjects))

            with self.subTest(f'pattern {i}, data type: raw'):
                _, y = self.loader.load(data_type=dtype, window_size=256, stride=256, ftrim_sec=5, btrim_sec=5, subjects=subjects)
                ## data check
                self.assertSetEqual(set(np.unique(y[:, 1])), set(subjects))
    

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    UniMibTest.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
