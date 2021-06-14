import sys
import unittest

import itertools
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.ucihar import UCIHAR, load, load_raw, load_meta, reformat


class UCIHARTest(unittest.TestCase):
    path = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = UCIHAR(self.path)

    def tearDown(self):
        pass

    def test_load_fn(self):
        for include_gravity in [True, False]:
            with self.subTest(f'include_gravity: {include_gravity}'):
                data, meta = load(self.path, include_gravity=include_gravity)

                # compare between data and meta
                self.assertEqual(len(data), len(meta))

                # meta
                ## type check
                self.assertIsInstance(meta, pd.DataFrame)

                ## shape and column check
                self.assertSetEqual(set(meta.columns), set(['activity', 'person_id', 'train']))

                ## data type check
                # flags_dtype = [dt == np.dtype(np.int8) or dt == np.dtype(np.int16) or dt == np.dtype(np.int32) or dt == np.dtype(np.int64) for dt in meta.dtypes[:-1]]
                flags_dtype = [dt == np.dtype(np.int8) for dt in meta.dtypes[:-1]]
                self.assertTrue(all(flags_dtype), meta.dtypes[:-1])
                self.assertTrue(meta.dtypes[-1] == np.dtype(bool), meta.dtypes[-1])

                ## data check
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 6+1)))
                self.assertSetEqual(set(np.unique(meta['person_id'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['train'])), set([0, 1]))

                # data
                ## type check
                self.assertIsInstance(data, list)
                self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

                ## shape check
                flgs_shape = [len(d.shape) == 2 for d in data]
                self.assertTrue(all(flgs_shape))
                flgs_shape_ax1 = [d.shape[1] == 3 for d in data]
                self.assertTrue(all(flgs_shape_ax1))
                self.assertTrue(all([
                    set(d.columns) == set(['x', 'y', 'z']) for d in data
                ]))

                ## data type check
                flags_dtype = list(itertools.chain([dt == np.dtype(np.float64) for dt in d.dtypes] for d in data))
                self.assertTrue(all(flags_dtype), data[0].dtypes[:-1])

                ## data check
                sizes_seg = set(len(d) for d in data)
                self.assertEqual(len(sizes_seg), 1)

    def test_load_raw_fn(self):
        for include_gravity in [True, False]:
            with self.subTest(f'include_gravity: {include_gravity}'):
                raw = load_raw(self.path, include_gravity=include_gravity)

                # raw
                ## type check
                self.assertIsInstance(raw, tuple)

                ## shape check
                self.assertEqual(len(raw), 2)

                data, meta = raw

                # compare between data and meta
                self.assertEqual(len(data), len(meta))

                # meta
                ## type check
                self.assertIsInstance(meta, pd.DataFrame)

                ## shape and column check
                self.assertSetEqual(set(meta.columns), set(['activity', 'person_id', 'train']))

                ## data type check
                # flags_dtype = [dt == np.dtype(np.int8) or dt == np.dtype(np.int16) or dt == np.dtype(np.int32) or dt == np.dtype(np.int64) for dt in meta.dtypes[:-1]]
                flags_dtype = [dt == np.dtype(np.int8) for dt in meta.dtypes[:-1]]
                self.assertTrue(all(flags_dtype), meta.dtypes[:-1])
                self.assertTrue(meta.dtypes[-1] == np.dtype(bool), meta.dtypes[-1])
                del flags_dtype

                ## data check
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 6+1)))
                self.assertSetEqual(set(np.unique(meta['person_id'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['train'])), set([0, 1]))

                # data
                ## type check
                self.assertIsInstance(data, np.ndarray)
                self.assertTrue(all(isinstance(d, np.ndarray) for d in data))

                ## data type check
                flags_dtype = [d.dtype == np.dtype(np.float64) for d in data]
                self.assertTrue(all(flags_dtype), data[0].dtype)

                ## shape check
                self.assertTupleEqual(data.shape[1:], (3, 128))
    
    def test_load_meta_fn(self):
        meta = load_meta(self.path)

        ## type check
        self.assertIsInstance(meta, pd.DataFrame)

        ## shape and column check
        self.assertSetEqual(set(meta.columns), set(['activity', 'person_id', 'train']))

        ## data type check
        # flags_dtype = [dt == np.dtype(np.int8) or dt == np.dtype(np.int16) or dt == np.dtype(np.int32) or dt == np.dtype(np.int64) for dt in meta.dtypes[:-1]]
        flags_dtype = [dt == np.dtype(np.int8) for dt in meta.dtypes[:-1]]
        self.assertTrue(all(flags_dtype), meta.dtypes[:-1])
        self.assertTrue(meta.dtypes[-1] == np.dtype(bool), meta.dtypes[-1])

        ## data check
        self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 6+1)))
        self.assertSetEqual(set(np.unique(meta['person_id'])), set(range(1, 30+1)))
        self.assertSetEqual(set(np.unique(meta['train'])), set([0, 1]))

    def test_reformat_fn(self):
        for include_gravity in [True, False]:
            with self.subTest(f'include_gravity: {include_gravity}'):
                raw = load_raw(self.path, include_gravity=include_gravity)
                data, meta = reformat(raw)

                # compare between data and meta
                self.assertEqual(len(data), len(meta))

                # meta
                ## type check
                self.assertIsInstance(meta, pd.DataFrame)

                ## shape and column check
                self.assertSetEqual(set(meta.columns), set(['activity', 'person_id', 'train']))

                ## data type check
                # flags_dtype = [dt == np.dtype(np.int8) or dt == np.dtype(np.int16) or dt == np.dtype(np.int32) or dt == np.dtype(np.int64) for dt in meta.dtypes[:-1]]
                flags_dtype = [dt == np.dtype(np.int8) for dt in meta.dtypes[:-1]]
                self.assertTrue(all(flags_dtype), meta.dtypes[:-1])
                self.assertTrue(meta.dtypes[-1] == np.dtype(bool), meta.dtypes[-1])

                ## data check
                self.assertSetEqual(set(np.unique(meta['activity'])), set(range(1, 6+1)))
                self.assertSetEqual(set(np.unique(meta['person_id'])), set(range(1, 30+1)))
                self.assertSetEqual(set(np.unique(meta['train'])), set([0, 1]))

                # data
                ## type check
                self.assertIsInstance(data, list)
                self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

                ## shape check
                flgs_shape = [len(d.shape) == 2 for d in data]
                self.assertTrue(all(flgs_shape))
                flgs_shape_ax1 = [d.shape[1] == 3 for d in data]
                self.assertTrue(all(flgs_shape_ax1))
                self.assertTrue(all([
                    set(d.columns) == set(['x', 'y', 'z']) for d in data
                ]))

                ## data type check
                flags_dtype = list(itertools.chain([dt == np.dtype(np.float64) for dt in d.dtypes] for d in data))
                self.assertTrue(all(flags_dtype), data[0].dtypes[:-1])

                ## data check
                sizes_seg = set(len(d) for d in data)
                self.assertEqual(len(sizes_seg), 1)

    def test_ucihar_load_method_base(self):
        # train=True, include_gravity=True
        x, y = self.loader.load(train=True, person_list=None, include_gravity=True)

        ## type check
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        ## shape check
        self.assertEqual(len(x.shape), 3)
        self.assertTupleEqual(x.shape[1:], (3, 128))
        self.assertEqual(len(y.shape), 2)
        self.assertEqual(y.shape[1], 3)

        ## data type check
        # self.assertTrue(x.dtype == np.dtype(np.float32) or x.dtype == np.dtype(np.float64))
        # self.assertTrue(y.dtype == np.dtype(np.int8) or y.dtype == np.dtype(np.int16) or y.dtype == np.dtype(np.int32) or y.dtype == np.dtype(np.int64))
        self.assertTrue(x.dtype == np.dtype(np.float64))
        self.assertTrue(y.dtype == np.dtype(np.int8))

        ## data check
        self.assertSetEqual(set(np.unique(y[:, 0])), set(range(6))) # activity
        self.assertSetEqual(set(np.unique(y[:, 1])), set([1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]))   # person_id
        self.assertSetEqual(set(np.unique(y[:, 2])), set([1]))   # train

        del x, y

        # train=True, include_gravity=False
        x, y = self.loader.load(train=True, person_list=None, include_gravity=False)

        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        self.assertEqual(len(x.shape), 3)
        self.assertTupleEqual(x.shape[1:], (3, 128))
        self.assertEqual(len(y.shape), 2)
        self.assertEqual(y.shape[1], 3)

        # self.assertTrue(x.dtype == np.dtype(np.float32) or x.dtype == np.dtype(np.float64))
        # self.assertTrue(y.dtype == np.dtype(np.int8) or y.dtype == np.dtype(np.int16) or y.dtype == np.dtype(np.int32) or y.dtype == np.dtype(np.int64))
        self.assertTrue(x.dtype == np.dtype(np.float64))
        self.assertTrue(y.dtype == np.dtype(np.int8))

        self.assertSetEqual(set(np.unique(y[:, 0])), set(range(6)))
        self.assertSetEqual(set(np.unique(y[:, 1])), set([1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]))   
        self.assertSetEqual(set(np.unique(y[:, 2])), set([1]))

        del x, y

        # train=False, include_gravity=True
        x, y = self.loader.load(train=False, person_list=None, include_gravity=True)

        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        self.assertEqual(len(x.shape), 3)
        self.assertTupleEqual(x.shape[1:], (3, 128))
        self.assertEqual(len(y.shape), 2)
        self.assertEqual(y.shape[1], 3)

        # self.assertTrue(x.dtype == np.dtype(np.float32) or x.dtype == np.dtype(np.float64))
        # self.assertTrue(y.dtype == np.dtype(np.int8) or y.dtype == np.dtype(np.int16) or y.dtype == np.dtype(np.int32) or y.dtype == np.dtype(np.int64))
        self.assertTrue(x.dtype == np.dtype(np.float64))
        self.assertTrue(y.dtype == np.dtype(np.int8))

        self.assertSetEqual(set(np.unique(y[:, 0])), set(range(6)))
        self.assertSetEqual(set(np.unique(y[:, 1])), set([2, 4, 9, 10, 12, 13, 18, 20, 24]))
        self.assertSetEqual(set(np.unique(y[:, 2])), set([0]))

        del x, y

        # train=False, include_gravity=False
        x, y = self.loader.load(train=False, person_list=None, include_gravity=False)

        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        self.assertEqual(len(x.shape), 3)
        self.assertTupleEqual(x.shape[1:], (3, 128))
        self.assertEqual(len(y.shape), 2)
        self.assertEqual(y.shape[1], 3)

        # self.assertTrue(x.dtype == np.dtype(np.float32) or x.dtype == np.dtype(np.float64))
        # self.assertTrue(y.dtype == np.dtype(np.int8) or y.dtype == np.dtype(np.int16) or y.dtype == np.dtype(np.int32) or y.dtype == np.dtype(np.int64))
        self.assertTrue(x.dtype == np.dtype(np.float64))
        self.assertTrue(y.dtype == np.dtype(np.int8))

        self.assertSetEqual(set(np.unique(y[:, 0])), set(range(6)))
        self.assertSetEqual(set(np.unique(y[:, 1])), set([2, 4, 9, 10, 12, 13, 18, 20, 24]))
        self.assertSetEqual(set(np.unique(y[:, 2])), set([0]))

        del x, y

    def test_ucihar_load_method_filed_subjects(self):
        def _check(train, patterns):
            for i, subjects in enumerate(patterns):
                for include_gravity in [True, False]:
                    with self.subTest(f'pattern {i}, train: {train}, include_gravity: {include_gravity}'):
                        _, y = self.loader.load(train=train, person_list=subjects, include_gravity=include_gravity)
                        ## data check
                        self.assertSetEqual(set(np.unique(y[:, 1])), set(subjects))

        # train=True
        patterns_train = [
            [30, 17, 23, 22, 3, 21, 19, 29, 15, 11, 26, 7, 5],
            [7, 21, 14, 27, 29, 11, 17, 23, 19, 3, 28, 25, 5, 1, 30, 6, 26, 16],
            [29],
            [3, 25],
            [7, 30, 11, 5, 19, 14, 17, 21, 27, 16, 26, 3, 1, 29, 8, 22, 28, 6, 25, 23, 15],
            [27, 21, 11, 7, 23, 30, 15, 3, 19, 25, 29, 28, 1, 16, 8, 14, 26],
            [8, 29, 22, 26],
            [11, 16, 3, 8, 23, 14, 26, 7, 17],
            [27, 25, 14, 5, 17, 30, 29, 7, 1, 15, 3, 6, 8, 11],
            [8, 26, 7, 1, 17, 30, 5, 6, 21, 27, 23, 3, 28, 22],
        ]

        _check(train=True, patterns=patterns_train)

        # trian=False
        patterns_test = [
            [9, 24, 12, 13, 4, 18],
            [10, 13, 4, 9, 12],
            [9, 10, 24, 12, 13, 4, 2, 18],
            [18, 4, 12, 9, 20],
            [13, 9, 20, 12],
            [4],
            [10],
            [2, 12, 9, 10, 13],
            [20, 24, 9],
            [4, 9, 13, 2, 24],
        ]

        _check(train=False, patterns=patterns_test)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    UCIHARTest.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
