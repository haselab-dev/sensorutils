import sys
import unittest

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.hhar import HHAR, load, load_raw, reformat
from sensorutils.datasets.hhar import ACTIVITIES, SUBJECTS, MODELS, PHONE_DEVICES, WATCH_DEVICES


class HHARTest(unittest.TestCase):
    path = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        self.loader = HHAR(self.path)

    def tearDown(self):
        pass

    def test_load_fn(self):
        for sensor, dev in itertools.product(['accelerometer', 'gyroscope'], ['Watch', 'Phone']):
            with self.subTest(f'sensor: {sensor}, device: {dev}'):
                data, meta = load(self.path, sensor_type=sensor, device_type=dev)

                # compare between data and meta
                self.assertEqual(len(data), len(meta))

                # check meta
                ## type check
                self.assertIsInstance(meta, list)
                self.assertTrue(all(isinstance(m, pd.DataFrame) for m in meta))

                ## shape and column check
                self.assertTrue(all([set(m.columns) == set(['Index', 'Arrival_Time', 'Creation_Time', 'User', 'Model', 'Device', 'gt']) for m in meta]))

                ## data type check
                self.assertTrue(all([
                    (np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.int64), np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.int8))== \
                    tuple(m.dtypes) for m in meta
                ]))

                ## data check
                self.assertTrue(all([
                    [set(np.unique(m['User'])) == set([m['User'].iloc[0]]) for m in meta]
                ]))
                self.assertTrue(all([
                    [set(np.unique(m['gt'])) == set([m['gt'].iloc[0]]) for m in meta]
                ]))
                self.assertTrue(all([
                    [set(np.unique(m['Device'])) == set([m['Device'].iloc[0]]) for m in meta]
                ]))

                # data
                ## type check
                self.assertIsInstance(data, list)
                self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

                ## shape and column check
                self.assertTrue(all([len(d.shape) == 2 for d in data]))
                self.assertTrue(all([d.shape[1] == 3 for d in data]))
                self.assertTrue(all([set(d.columns) == set(['x', 'y', 'z']) for d in data]))

                ## data type check
                self.assertTrue(all([
                    (np.dtype(np.float64), np.dtype(np.float64), np.dtype(np.float64))== \
                    tuple(d.dtypes) for d in data
                ]))

                del data, meta

    def test_load_raw_fn(self):
        for sensor, dev in itertools.product(['accelerometer', 'gyroscope'], ['Watch', 'Phone']):
            with self.subTest(f'sensor: {sensor}, device: {dev}'):
                raw = load_raw(self.path, sensor_type=sensor, device_type=dev)

                # raw
                ## type check
                self.assertIsInstance(raw, pd.DataFrame)

                ## shape and column check
                self.assertSetEqual(set(raw.columns), set(['Index', 'Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User','Model', 'Device', 'gt']))

                ## data type check
                self.assertTupleEqual(
                    tuple(raw.dtypes),
                    (np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.int64), np.dtype(np.float64), np.dtype(np.float64), np.dtype(np.float64), np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.int8))
                )

                ## data check
                self.assertSetEqual(set(np.unique(raw['User'])), set(SUBJECTS.values())) # subject 
                if dev == 'Watch':
                    self.assertSetEqual(set(np.unique(raw['Model'])), set([4, 5])) # model 
                elif dev == 'Phone':
                    if sensor == 'accelerometer':
                        self.assertSetEqual(set(np.unique(raw['Model'])), set([0, 1, 2, 3])) # model 
                    elif sensor == 'gyroscope':
                        # samsungold(3) not found
                        self.assertSetEqual(set(np.unique(raw['Model'])), set([0, 1, 2])) # model 
                    else:
                        raise RuntimeError
                else:
                    raise RuntimeError
                if dev == 'Watch':
                    self.assertSetEqual(set(np.unique(raw['Device'])), set(WATCH_DEVICES.values())) # device
                elif dev == 'Phone':
                    if sensor == 'accelerometer':
                        self.assertSetEqual(set(np.unique(raw['Device'])), set(PHONE_DEVICES.values())) # device
                    elif sensor == 'gyroscope':
                        # samsungold(6, 7) not found
                        self.assertSetEqual(set(np.unique(raw['Device'])), set([0, 1, 2, 3, 4, 5])) # device
                    else:
                        raise RuntimeError
                else:
                    raise RuntimeError
                self.assertSetEqual(set(np.unique(raw['gt'])), set(ACTIVITIES.values())) # gt(activity)

                del raw

    def test_reformat_fn(self):
        for sensor, dev in itertools.product(['accelerometer', 'gyroscope'], ['Watch', 'Phone']):
            with self.subTest(f'sensor: {sensor}, device: {dev}'):
                raw = load_raw(self.path, sensor_type=sensor, device_type=dev)
                data, meta = reformat(raw)

                # compare between data and meta
                self.assertEqual(len(data), len(meta))

                # check meta
                ## type check
                self.assertIsInstance(meta, list)
                self.assertTrue(all(isinstance(m, pd.DataFrame) for m in meta))

                ## shape and column check
                self.assertTrue(all([set(m.columns) == set(['Index', 'Arrival_Time', 'Creation_Time', 'User', 'Model', 'Device', 'gt']) for m in meta]))

                ## data type check
                self.assertTrue(all([
                    (np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.int64), np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.int8))== \
                    tuple(m.dtypes) for m in meta
                ]))

                ## data check
                self.assertTrue(all([
                    [set(np.unique(m['User'])) == set([m['User'].iloc[0]]) for m in meta]
                ]))
                self.assertTrue(all([
                    [set(np.unique(m['gt'])) == set([m['gt'].iloc[0]]) for m in meta]
                ]))
                self.assertTrue(all([
                    [set(np.unique(m['Device'])) == set([m['Device'].iloc[0]]) for m in meta]
                ]))

                # data
                ## type check
                self.assertIsInstance(data, list)
                self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

                ## shape and column check
                self.assertTrue(all([len(d.shape) == 2 for d in data]))
                self.assertTrue(all([d.shape[1] == 3 for d in data]))
                self.assertTrue(all([set(d.columns) == set(['x', 'y', 'z']) for d in data]))

                ## data type check
                self.assertTrue(all([
                    (np.dtype(np.float64), np.dtype(np.float64), np.dtype(np.float64))== \
                    tuple(d.dtypes) for d in data
                ]))

                del data, meta

    def test_hhar_load_method_framing(self):
        for sensor, dev, stride, ws in itertools.product(['accelerometer', 'gyroscope'], ['Watch', 'Phone'], [256, 512], [256, 512]):
            with self.subTest(f'sensor: {sensor}, device: {dev}, window size: {ws}, stride: {stride}'):
                x, y = self.loader.load(sensor_types=sensor, device_types=dev, window_size=ws, stride=stride, subjects=None)

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
                self.assertEqual(y.shape[1], 4)

                ## data check
                self.assertSetEqual(set(np.unique(y[:, 0])), set(SUBJECTS.values())) # subject 
                if dev == 'Watch':
                    self.assertSetEqual(set(np.unique(y[:, 1])), set([4, 5])) # model 
                elif dev == 'Phone':
                    if sensor == 'accelerometer':
                        self.assertSetEqual(set(np.unique(y[:, 1])), set([0, 1, 2, 3])) # model 
                    elif sensor == 'gyroscope':
                        self.assertSetEqual(set(np.unique(y[:, 1])), set([0, 1, 2])) # model 
                    else:
                        raise RuntimeError
                else:
                    raise RuntimeError
                if dev == 'Watch':
                    self.assertSetEqual(set(np.unique(y[:, 2])), set(WATCH_DEVICES.values())) # device
                elif dev == 'Phone':
                    if sensor == 'accelerometer':
                        self.assertSetEqual(set(np.unique(y[:, 2])), set(PHONE_DEVICES.values())) # device
                    elif sensor == 'gyroscope':
                        self.assertSetEqual(set(np.unique(y[:, 2])), set([0, 1, 2, 3, 4, 5])) # device
                    else:
                        raise RuntimeError
                else:
                    raise RuntimeError
                ### activity=nullのデータが除去されていることも確認
                self.assertSetEqual(set(np.unique(y[:, 3])), set(ACTIVITIES.values())-set([ACTIVITIES['null']])) # gt(activity)

                del x, y

    def test_hhar_load_method_filed_subjects(self):
        patterns = [
            ['c', 'i', 'e', 'f', 'b', 'g'],
            ['d', 'f', 'b', 'c', 'e'],
            ['c', 'd', 'i', 'e', 'f', 'b', 'a', 'g'],
            ['g', 'b', 'e', 'c', 'h'],
            ['f', 'c', 'h', 'e'],
            ['b'],
            ['d'],
            ['a', 'e', 'c', 'd', 'f'],
            ['h', 'i', 'c'],
            ['b', 'c', 'f', 'a', 'i'],
        ]

        for i, subjects in enumerate(patterns):
            with self.subTest(f'pattern {i}'):
                _, y = self.loader.load(sensor_types='accelerometer', device_types='Watch', window_size=256, stride=256, subjects=subjects)
                S = list(map(lambda x: SUBJECTS[x], subjects))
                self.assertSetEqual(set(np.unique(y[:, 0])), set(S))


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    HHARTest.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
