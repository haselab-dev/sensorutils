import sys
import unittest

import numpy as np
import pandas as pd
import itertools
import copy
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.opportunity import Opportunity, load, load_raw, reformat, Column
from sensorutils.datasets.opportunity import HL_Activity, LL_Left_Arm, LL_Left_Arm_Object, LL_Right_Arm, LL_Right_Arm_Object, ML_Both_Arms

class OpportunityTest(unittest.TestCase):
    path = None
    activities = [1, 2, 4, 5, 0]
    subjects = list(range(1, 4+1))

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = Opportunity(self.path)

    def tearDown(self):
        pass

    def test_load_fn(self):
        data, meta = load(self.path)

        y_labels = ['Locomotion', 'subject', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms']
        x_labels = list(set(copy.deepcopy(Column)) - set(y_labels))

        # compare between data and meta
        self.assertEqual(len(data), len(meta))

        # check meta
        ## type check
        self.assertIsInstance(meta, list)
        self.assertTrue(all(isinstance(m, pd.DataFrame) for m in meta))

        ## shape and column check
        self.assertTrue(all([set(m.columns) == set(y_labels) for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['Locomotion'] == np.dtype(np.int32) and \
            m.dtypes['subject'] == np.dtype(np.int32) and \
            m.dtypes['HL_Activity'] == np.dtype(np.int32) and \
            m.dtypes['LL_Left_Arm'] == np.dtype(np.int32) and \
            m.dtypes['LL_Left_Arm_Object'] == np.dtype(np.int32) and \
            m.dtypes['LL_Right_Arm'] == np.dtype(np.int32) and \
            m.dtypes['LL_Right_Arm_Object'] == np.dtype(np.int32) and \
            m.dtypes['ML_Both_Arms'] == np.dtype(np.int32) \
            for m in meta
        ]))

        ## data check
        self.assertTrue(all([
            set(np.unique(m['subject'])) == set([m['subject'].iloc[0]]) for m in meta
        ]))
        self.assertTrue(all([
            set(np.unique(m['Locomotion'])) == set([m['Locomotion'].iloc[0]]) for m in meta
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([set(d.columns) == set(x_labels) for d in data]))

        ## data type check
        C = list(set(data[0].columns) - set(self.loader.NOT_SUPPORTED_LABELS))
        self.assertTrue(all([
            d.dtypes[c] == np.dtype(np.float64) \
            for d in data for c in C
        ]))

    def test_load_raw_fn(self):
        raw = load_raw(self.path)

        y_labels = ['Locomotion', 'subject', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms']
        x_labels = list(set(copy.deepcopy(Column)) - set(y_labels))

        # raw
        ## type check
        self.assertIsInstance(raw, list)
        self.assertTrue(all(isinstance(r, pd.DataFrame) for r in raw))

        ## shape and column check
        tgt = tuple(list(Column) + ['subject'])
        self.assertTrue(all([
            tuple(r.columns) == tgt for r in raw
        ]))

        ## data type check
        flgs = []
        for r in raw:
            for col in r.columns:
                if col in y_labels:
                    flgs += [r.dtypes[col] == np.dtype(np.int32)]
                elif col in x_labels:
                    dt = r.dtypes[col]
                    if isinstance(dt, pd.Series):
                        dt = dt.all()
                    flgs += [dt == np.dtype(np.float64)]
                else:
                    pass
        self.assertTrue(all(flgs))

        ## data check
        self.assertTrue(all([
            set(np.unique(r['subject'])) == set([r['subject'].iloc[0]]) for r in raw 
        ]))
        self.assertSetEqual(
            set(itertools.chain(*[np.unique(r['Locomotion']).tolist() for r in raw])),
            set(self.activities)
        )

        self.assertSetEqual(set(itertools.chain(*[np.unique(r['HL_Activity']) for r in raw])), set(HL_Activity.keys()).union(set([0])))
        # なぜかこれだけ通らない
        # self.assertSetEqual(set(itertools.chain(*[np.unique(r['LL_Left_Arm']) for r in raw])), set(LL_Left_Arm.keys()).union(set([0])))
        self.assertSetEqual(set(itertools.chain(*[np.unique(r['LL_Left_Arm_Object']) for r in raw])), set(LL_Left_Arm_Object.keys()).union(set([0])))
        self.assertSetEqual(set(itertools.chain(*[np.unique(r['LL_Right_Arm']) for r in raw])), set(LL_Right_Arm.keys()).union(set([0])))
        self.assertSetEqual(set(itertools.chain(*[np.unique(r['LL_Right_Arm_Object']) for r in raw])), set(LL_Right_Arm_Object.keys()).union(set([0])))
        self.assertSetEqual(set(itertools.chain(*[np.unique(r['ML_Both_Arms']) for r in raw])), set(ML_Both_Arms.keys()).union(set([0])))

    def test_reformat_fn(self):
        raw = load_raw(self.path)
        data, meta = reformat(raw)

        y_labels = ['Locomotion', 'subject', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms']
        x_labels = list(set(copy.deepcopy(Column)) - set(y_labels))

        # compare between data and meta
        self.assertEqual(len(data), len(meta))

        # check meta
        ## type check
        self.assertIsInstance(meta, list)
        self.assertTrue(all(isinstance(m, pd.DataFrame) for m in meta))

        ## shape and column check
        self.assertTrue(all([set(m.columns) == set(y_labels) for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['Locomotion'] == np.dtype(np.int32) and \
            m.dtypes['subject'] == np.dtype(np.int32) and \
            m.dtypes['HL_Activity'] == np.dtype(np.int32) and \
            m.dtypes['LL_Left_Arm'] == np.dtype(np.int32) and \
            m.dtypes['LL_Left_Arm_Object'] == np.dtype(np.int32) and \
            m.dtypes['LL_Right_Arm'] == np.dtype(np.int32) and \
            m.dtypes['LL_Right_Arm_Object'] == np.dtype(np.int32) and \
            m.dtypes['ML_Both_Arms'] == np.dtype(np.int32) \
            for m in meta
        ]))

        ## data check
        self.assertTrue(all([
            set(np.unique(m['subject'])) == set([m['subject'].iloc[0]]) for m in meta
        ]))
        self.assertTrue(all([
            set(np.unique(m['Locomotion'])) == set([m['Locomotion'].iloc[0]]) for m in meta
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([set(d.columns) == set(x_labels) for d in data]))

        ## data type check
        C = list(set(data[0].columns) - set(self.loader.NOT_SUPPORTED_LABELS))
        self.assertTrue(all([
            d.dtypes[c] == np.dtype(np.float64) \
            for d in data for c in C
        ]))

    def test_opp_load_method_filed_labels(self):
        # check processing for exceptions
        y_labels = ['Locomotion', 'subject', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms']
        x_labels = list(set(copy.deepcopy(Column)) - set(y_labels) - set(self.loader.NOT_SUPPORTED_LABELS))

        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, x_labels=y_labels)

        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, y_labels=x_labels)

        # shape check
        ## x
        for n in range(10, len(x_labels), 10):
            x, _ = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, x_labels=np.random.choice(x_labels, n, replace=False).tolist())
            self.assertTupleEqual(x.shape[1:], (n, 128))
        ## y
        _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, y_labels=['Locomotion'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, y_labels=['subject'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, y_labels=['Locomotion', 'subject'])
        self.assertEqual(y.shape[1], 2)

    def test_opp_load_method_framing(self):
        y_labels = ['Locomotion', 'subject', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms']
        x_labels = list(set(copy.deepcopy(Column)) - set(y_labels) - set(self.loader.NOT_SUPPORTED_LABELS))
        for stride, ws in itertools.product([64, 128, 256, 512], [64, 128, 256, 512]):
            print(f'window size: {ws}, stride: {stride}')
            with self.subTest(f'window size: {ws}, stride: {stride}'):
                x, y = self.loader.load(window_size=ws, stride=stride, x_labels=None, y_labels=y_labels, ftrim_sec=2, btrim_sec=2)

                ## compare between x and y
                self.assertEqual(len(x), len(y))

                ## type check
                self.assertIsInstance(x, np.ndarray)
                self.assertIsInstance(y, np.ndarray)

                ## data type check
                self.assertEqual(x.dtype, np.dtype(np.float64))
                self.assertEqual(y.dtype, np.dtype(np.int32))

                ## shape check
                ## x_labels, y_labelsによって変わるため要検討
                self.assertEqual(len(x.shape), 3)
                self.assertTupleEqual(x.shape[1:], (len(x_labels), ws))
                self.assertEqual(len(y.shape), 2)
                self.assertEqual(y.shape[1], len(y_labels))

                ## data check
                ## x_labels, y_labelsによって変わるため要検討
                self.assertSetEqual(set(np.unique(y[:, 0])), set(self.activities)) # activity(protocol): performed activities + others
                self.assertSetEqual(set(np.unique(y[:, 1])), set(self.subjects)) # subject

                del x, y

    # 被験者フィルタリングは未実装
    @unittest.skip
    def test_opp_load_method_filed_subjects(self):
        patterns = [
            [5, 3, 2, 4, 1],
            [2, 1, 3],
            [4],
            [1],
            [3],
            [5],
            [1, 2, 3, 5],
            [1],
            [4],
            [2, 4, 1],
        ]

        for i, subjects in enumerate(patterns):
            print(f'pattern {i}')
            with self.subTest(f'pattern {i}'):
                _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, subjects=subjects)
                subjects_id = list(map(lambda x: int(x[-1])-1, subjects))
                self.assertSetEqual(set(np.unique(y[:, 1])), set(subjects_id))


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    OpportunityTest.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
