from os import replace
import sys
import unittest

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.pamap2 import PAMAP2, load, load_raw, reformat


class PAMAP2Test(unittest.TestCase):
    path = None
    protocol_activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24, 0]
    optional_activities = [9, 10, 11, 18, 19, 20, 0]
    subjects = list(range(9))

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = PAMAP2(self.path)

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
        self.assertTrue(all([set(m.columns) == set(['activity_id', 'person_id']) for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['activity_id'] == np.dtype(np.int8) and \
            m.dtypes['person_id'] == np.dtype(np.int8) \
            for m in meta
        ]))

        ## data check
        self.assertTrue(all([
            set(np.unique(m['person_id'])) == set([m['person_id'].iloc[0]]) for m in meta
        ]))
        self.assertTrue(all([
            set(np.unique(m['activity_id'])) == set([m['activity_id'].iloc[0]]) for m in meta
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([
            set(d.columns) == \
            set(['timestamp(s)', 'heart_rate(bpm)', 'IMU_hand_temperature', 'IMU_hand_acc1_x', 'IMU_hand_acc1_y', 'IMU_hand_acc1_z', 'IMU_hand_acc2_x', 'IMU_hand_acc2_y', 'IMU_hand_acc2_z', 'IMU_hand_gyro_x', 'IMU_hand_gyro_y', 'IMU_hand_gyro_z', 'IMU_hand_mag_x', 'IMU_hand_mag_y', 'IMU_hand_mag_z', 'IMU_hand_orientation0', 'IMU_hand_orientation1', 'IMU_hand_orientation2', 'IMU_hand_orientation3', 'IMU_chest_temperature', 'IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z', 'IMU_chest_acc2_x', 'IMU_chest_acc2_y', 'IMU_chest_acc2_z', 'IMU_chest_gyro_x', 'IMU_chest_gyro_y', 'IMU_chest_gyro_z', 'IMU_chest_mag_x', 'IMU_chest_mag_y', 'IMU_chest_mag_z', 'IMU_chest_orientation0', 'IMU_chest_orientation1', 'IMU_chest_orientation2', 'IMU_chest_orientation3', 'IMU_ankle_temperature', 'IMU_ankle_acc1_x', 'IMU_ankle_acc1_y', 'IMU_ankle_acc1_z', 'IMU_ankle_acc2_x', 'IMU_ankle_acc2_y', 'IMU_ankle_acc2_z', 'IMU_ankle_gyro_x', 'IMU_ankle_gyro_y', 'IMU_ankle_gyro_z', 'IMU_ankle_mag_x', 'IMU_ankle_mag_y', 'IMU_ankle_mag_z', 'IMU_ankle_orientation0', 'IMU_ankle_orientation1', 'IMU_ankle_orientation2', 'IMU_ankle_orientation3']) \
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
            'timestamp(s)', 'activity_id', 'heart_rate(bpm)',
            'IMU_hand_temperature', 'IMU_hand_acc1_x', 'IMU_hand_acc1_y', 'IMU_hand_acc1_z',
            'IMU_hand_acc2_x', 'IMU_hand_acc2_y', 'IMU_hand_acc2_z',
            'IMU_hand_gyro_x', 'IMU_hand_gyro_y', 'IMU_hand_gyro_z',
            'IMU_hand_mag_x', 'IMU_hand_mag_y', 'IMU_hand_mag_z',
            'IMU_hand_orientation0', 'IMU_hand_orientation1', 'IMU_hand_orientation2', 'IMU_hand_orientation3',
            'IMU_chest_temperature', 'IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z',
            'IMU_chest_acc2_x', 'IMU_chest_acc2_y', 'IMU_chest_acc2_z',
            'IMU_chest_gyro_x', 'IMU_chest_gyro_y', 'IMU_chest_gyro_z',
            'IMU_chest_mag_x', 'IMU_chest_mag_y', 'IMU_chest_mag_z',
            'IMU_chest_orientation0', 'IMU_chest_orientation1', 'IMU_chest_orientation2', 'IMU_chest_orientation3',
            'IMU_ankle_temperature', 'IMU_ankle_acc1_x', 'IMU_ankle_acc1_y', 'IMU_ankle_acc1_z',
            'IMU_ankle_acc2_x', 'IMU_ankle_acc2_y', 'IMU_ankle_acc2_z',
            'IMU_ankle_gyro_x','IMU_ankle_gyro_y', 'IMU_ankle_gyro_z',
            'IMU_ankle_mag_x', 'IMU_ankle_mag_y', 'IMU_ankle_mag_z',
            'IMU_ankle_orientation0', 'IMU_ankle_orientation1', 'IMU_ankle_orientation2', 'IMU_ankle_orientation3',
            'person_id', 
        ])
        self.assertTrue(all([
            tuple(r.columns) == tgt for r in raw
        ]))

        ## data type check
        flgs = []
        for r in raw:
            for col in r.columns:
                if col in ['activity_id', 'person_id']:
                    flgs += [r.dtypes[col] == np.dtype(np.int8)]
                else:
                    flgs += [r.dtypes[col] == np.dtype(np.float64)]
        self.assertTrue(all(flgs))

        ## data check
        self.assertTrue(all([
            set(np.unique(r['person_id'])) == set([r['person_id'].iloc[0]]) for r in raw 
        ]))
        self.assertSetEqual(
            set(itertools.chain(*[np.unique(r['activity_id']).tolist() for r in raw])),
            set(self.protocol_activities)    # performed activity(protocol) + others
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
        self.assertTrue(all([set(m.columns) == set(['activity_id', 'person_id']) for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['activity_id'] == np.dtype(np.int8) and \
            m.dtypes['person_id'] == np.dtype(np.int8) \
            for m in meta
        ]))

        ## data check
        self.assertTrue(all([
            set(np.unique(m['person_id'])) == set([m['person_id'].iloc[0]]) for m in meta
        ]))
        self.assertTrue(all([
            set(np.unique(m['activity_id'])) == set([m['activity_id'].iloc[0]]) for m in meta
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([
            set(d.columns) == \
            set(['timestamp(s)', 'heart_rate(bpm)', 'IMU_hand_temperature', 'IMU_hand_acc1_x', 'IMU_hand_acc1_y', 'IMU_hand_acc1_z', 'IMU_hand_acc2_x', 'IMU_hand_acc2_y', 'IMU_hand_acc2_z', 'IMU_hand_gyro_x', 'IMU_hand_gyro_y', 'IMU_hand_gyro_z', 'IMU_hand_mag_x', 'IMU_hand_mag_y', 'IMU_hand_mag_z', 'IMU_hand_orientation0', 'IMU_hand_orientation1', 'IMU_hand_orientation2', 'IMU_hand_orientation3', 'IMU_chest_temperature', 'IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z', 'IMU_chest_acc2_x', 'IMU_chest_acc2_y', 'IMU_chest_acc2_z', 'IMU_chest_gyro_x', 'IMU_chest_gyro_y', 'IMU_chest_gyro_z', 'IMU_chest_mag_x', 'IMU_chest_mag_y', 'IMU_chest_mag_z', 'IMU_chest_orientation0', 'IMU_chest_orientation1', 'IMU_chest_orientation2', 'IMU_chest_orientation3', 'IMU_ankle_temperature', 'IMU_ankle_acc1_x', 'IMU_ankle_acc1_y', 'IMU_ankle_acc1_z', 'IMU_ankle_acc2_x', 'IMU_ankle_acc2_y', 'IMU_ankle_acc2_z', 'IMU_ankle_gyro_x', 'IMU_ankle_gyro_y', 'IMU_ankle_gyro_z', 'IMU_ankle_mag_x', 'IMU_ankle_mag_y', 'IMU_ankle_mag_z', 'IMU_ankle_orientation0', 'IMU_ankle_orientation1', 'IMU_ankle_orientation2', 'IMU_ankle_orientation3']) \
            for d in data
        ]))

        ## data type check
        self.assertTrue(all([
            d.dtypes[c] == np.dtype(np.float64) \
            for d in data for c in data[0].columns
        ]))

    def test_pamap2_load_method_filed_labels(self):
        # check processing for exceptions
        x_labels = [
            'timestamp(s)', 'heart_rate(bpm)',
            'IMU_hand_temperature', 'IMU_hand_acc1_x', 'IMU_hand_acc1_y', 'IMU_hand_acc1_z',
            'IMU_hand_acc2_x', 'IMU_hand_acc2_y', 'IMU_hand_acc2_z',
            'IMU_hand_gyro_x', 'IMU_hand_gyro_y', 'IMU_hand_gyro_z',
            'IMU_hand_mag_x', 'IMU_hand_mag_y', 'IMU_hand_mag_z',
            'IMU_hand_orientation0', 'IMU_hand_orientation1', 'IMU_hand_orientation2', 'IMU_hand_orientation3',
            'IMU_chest_temperature', 'IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z',
            'IMU_chest_acc2_x', 'IMU_chest_acc2_y', 'IMU_chest_acc2_z',
            'IMU_chest_gyro_x', 'IMU_chest_gyro_y', 'IMU_chest_gyro_z',
            'IMU_chest_mag_x', 'IMU_chest_mag_y', 'IMU_chest_mag_z',
            'IMU_chest_orientation0', 'IMU_chest_orientation1', 'IMU_chest_orientation2', 'IMU_chest_orientation3',
            'IMU_ankle_temperature', 'IMU_ankle_acc1_x', 'IMU_ankle_acc1_y', 'IMU_ankle_acc1_z',
            'IMU_ankle_acc2_x', 'IMU_ankle_acc2_y', 'IMU_ankle_acc2_z',
            'IMU_ankle_gyro_x','IMU_ankle_gyro_y', 'IMU_ankle_gyro_z',
            'IMU_ankle_mag_x', 'IMU_ankle_mag_y', 'IMU_ankle_mag_z',
            'IMU_ankle_orientation0', 'IMU_ankle_orientation1', 'IMU_ankle_orientation2', 'IMU_ankle_orientation3'
        ]
        y_labels = ['activity_id', 'person_id']

        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, persons=None, x_labels=y_labels)

        with self.assertRaises(ValueError):
            _, _ = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, persons=None, y_labels=x_labels)

        # shape check
        ## x
        for n in range(10, len(x_labels), 10):
            x, _ = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, persons=None, x_labels=np.random.choice(x_labels, n, replace=False).tolist())
            self.assertTupleEqual(x.shape[1:], (n, 128))
        ## y
        _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, persons=None, y_labels=['activity_id'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, persons=None, y_labels=['person_id'])
        self.assertEqual(y.shape[1], 1)
        _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, persons=None, y_labels=['activity_id', 'person_id'])
        self.assertEqual(y.shape[1], 2)

    def test_pamap2_load_method_framing(self):
        x_labels = [
            'timestamp(s)', 'heart_rate(bpm)',
            'IMU_hand_temperature', 'IMU_hand_acc1_x', 'IMU_hand_acc1_y', 'IMU_hand_acc1_z',
            'IMU_hand_acc2_x', 'IMU_hand_acc2_y', 'IMU_hand_acc2_z',
            'IMU_hand_gyro_x', 'IMU_hand_gyro_y', 'IMU_hand_gyro_z',
            'IMU_hand_mag_x', 'IMU_hand_mag_y', 'IMU_hand_mag_z',
            'IMU_hand_orientation0', 'IMU_hand_orientation1', 'IMU_hand_orientation2', 'IMU_hand_orientation3',
            'IMU_chest_temperature', 'IMU_chest_acc1_x', 'IMU_chest_acc1_y', 'IMU_chest_acc1_z',
            'IMU_chest_acc2_x', 'IMU_chest_acc2_y', 'IMU_chest_acc2_z',
            'IMU_chest_gyro_x', 'IMU_chest_gyro_y', 'IMU_chest_gyro_z',
            'IMU_chest_mag_x', 'IMU_chest_mag_y', 'IMU_chest_mag_z',
            'IMU_chest_orientation0', 'IMU_chest_orientation1', 'IMU_chest_orientation2', 'IMU_chest_orientation3',
            'IMU_ankle_temperature', 'IMU_ankle_acc1_x', 'IMU_ankle_acc1_y', 'IMU_ankle_acc1_z',
            'IMU_ankle_acc2_x', 'IMU_ankle_acc2_y', 'IMU_ankle_acc2_z',
            'IMU_ankle_gyro_x','IMU_ankle_gyro_y', 'IMU_ankle_gyro_z',
            'IMU_ankle_mag_x', 'IMU_ankle_mag_y', 'IMU_ankle_mag_z',
            'IMU_ankle_orientation0', 'IMU_ankle_orientation1', 'IMU_ankle_orientation2', 'IMU_ankle_orientation3'
        ]
        y_labels = ['activity_id', 'person_id']
        for stride, ws in itertools.product([64, 128, 256, 512], [64, 128, 256, 512]):
            print(f'window size: {ws}, stride: {stride}')
            with self.subTest(f'window size: {ws}, stride: {stride}'):
                x, y = self.loader.load(window_size=ws, stride=stride, x_labels=None, y_labels=y_labels, ftrim_sec=2, btrim_sec=2, persons=None, norm=False)

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
                ## x_labels, y_labelsによって変わるため要検討
                self.assertSetEqual(set(np.unique(y[:, 0])), set(self.protocol_activities)) # activity(protocol): performed activities + others
                self.assertSetEqual(set(np.unique(y[:, 1])), set(range(9))) # subject

                del x, y

    def test_pamap2_load_method_filed_subjects(self):
        patterns = [
            ['subject103','subject109','subject105','subject106','subject102','subject107'],
            ['subject104','subject106','subject102','subject103','subject105'],
            ['subject103','subject104','subject109','subject105','subject106','subject102','subject101','subject107'],
            ['subject107','subject102','subject105','subject103','subject108'],
            ['subject106','subject103','subject108','subject105'],
            ['subject102'],
            ['subject104'],
            ['subject101','subject105','subject103','subject104','subject106'],
            ['subject108','subject109','subject103'],
            ['subject102','subject103','subject106','subject101','subject109'],
        ]

        for i, subjects in enumerate(patterns):
            print(f'pattern {i}')
            with self.subTest(f'pattern {i}'):
                _, y = self.loader.load(window_size=128, stride=128, ftrim_sec=2, btrim_sec=2, persons=subjects)
                subjects_id = list(map(lambda x: int(x[-1])-1, subjects))
                self.assertSetEqual(set(np.unique(y[:, 1])), set(subjects_id))


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    PAMAP2Test.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
