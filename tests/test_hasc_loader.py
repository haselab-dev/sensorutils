import sys
import unittest

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.hasc import HASC, load, load_raw, load_meta, reformat


class HASCTest(unittest.TestCase):
    path = None
    cache_path = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')
        if cls.cache_path is None:
            raise RuntimeError('dataset cache path is not specified')

    def setUp(self):
        self.loader = HASC(self.path, self.cache_path)

    def tearDown(self):
        pass

    @classmethod
    def _gen_small_meta(cls, meta):
        # filed_meta = meta.query('Frequency == 100 & Person in ["person01068", "person03053", "person02033", "person01106", "person03079"] & Height > 170')
        filed_meta = meta.query('Person in ["person01068", "person03053", "person02033", "person01106", "person03079"]')
        return filed_meta

    def test_load_fn(self):
        data, meta = load(self.path, meta=self._gen_small_meta(self.loader.meta.copy()))
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))
        self.assertIsInstance(meta, pd.DataFrame)
        self.assertEqual(len(data), len(meta))
    
    def test_load_raw_fn_base(self):
        meta_src = self._gen_small_meta(self.loader.meta.copy())
        raw = load_raw(self.path, meta=meta_src)
        self.assertIsInstance(raw, tuple)
        self.assertEqual(len(raw), 2)
        data, meta = raw
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))
        self.assertIsInstance(meta, pd.DataFrame)
        self.assertEqual(len(data), len(meta))
        self.assertTrue(meta_src.equals(meta))

    @unittest.skip
    def test_load_meta_fn(self):
        meta = load_meta(self.path)
        self.assertIsInstance(meta, pd.DataFrame)

    @unittest.skip
    def test_reformat_fn(self):
        data, meta = load_raw(self.path, meta=self._gen_small_meta(self.loader.meta.copy()))
        data, _ = reformat(data, meta)
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))
        self.assertIsInstance(meta, pd.DataFrame)
        self.assertEqual(len(data), len(meta))
    
    @unittest.skip
    def test_hasc_meta_map(self):
        _ = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, y_labels=['activity', 'frequency', 'gender', 'height', 'weight', 'person'])
        ...
    
    def test_hasc_load_method_base(self):
        # subjects = ['person01068', 'person03053', 'person02033', 'person01106', 'person03079']
        # queries = {'Person': 'Person in {}'.format(subjects)}
        queries = None
        y_labels = ['activity', 'person']
        _, _, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=y_labels)
        self.assertIsInstance(label_map, dict)
        self.assertTrue(set(y_labels), set(label_map.keys()))
        for k in label_map:
            M = label_map[k]
            if bool(M):
                with self.subTest(f'key of label map: {k}'):
                    self.assertSetEqual(set(M.values()), set(range(min(list(M.values())), max(list(M.values()))+1)))
    
    def test_hasc_load_method_framing(self):
        subjects = ['person01068', 'person03053', 'person02033', 'person01106', 'person03079']
        queries = {'Person': 'Person in {}'.format(subjects)}
        y_labels = ['activity', 'person']

        for ws in [128, 256, 512]:
            x, _, _= self.loader.load(window_size=ws, stride=ws, ftrim=5, btrim=5, queries=queries, y_labels=y_labels)
            with self.subTest(f'window_size: {ws}'):
                self.assertTupleEqual(x.shape[1:], (3, ws))
    
    def test_hasc_load_method_ylabels(self):
        subjects = ['person01068', 'person03053', 'person02033', 'person01106', 'person03079']
        queries = {'Person': 'Person in {}'.format(subjects)}
        y_labels = ['activity', 'frequency', 'gender', 'height', 'weight', 'person']

        for n in range(1, len(y_labels)):
            with self.subTest(f'number of y labels: {n}'):
                Y = np.random.choice(y_labels, n)
                _, y, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=Y)
                self.assertIsInstance(label_map, dict)
                self.assertTrue(set(y_labels), set(label_map.keys()))
                self.assertEqual(y.shape[1], n)

    def test_hasc_load_method_filed_meta(self):
        subjects = ['person01068', 'person03053', 'person02033', 'person01106', 'person03079', 'person02007', 'person01085', 'person01060', 'person01103', 'person03032', 'person01107', 'person01045', 'person02063', 'person03055', 'person01066', 'person03001', 'person01039', 'person01113', 'person03034', 'person03056', 'person02100', 'person01087', 'person01089', 'person01109', 'person01017', 'person01063', 'person01098', 'person03038', 'person02012', 'person01097', 'person03036', 'person03033', 'person01078', 'person02068', 'person03076', 'person01040', 'person02024', 'person01073', 'person02040']
        with self.subTest('filtered by activity'):
            queries = {'Person': 'Person in {}'.format(subjects)}
            _, y, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=['activity', 'frequency', 'gender', 'height', 'weight', 'person'])
            self.assertIsInstance(label_map, dict)
            self.assertSetEqual(set(label_map['activity'].keys()), set(['1_stay', '2_walk', '3_jog', '4_skip', '5_stUp', '6_stDown']))
            self.assertEqual(y[:, 0].min(), 0)
            self.assertEqual(y[:, 0].max(), 5)

        g = 'male'
        with self.subTest(f'filtered by gender(gender == "{g}")'):
            queries = {'Gender': f'Gender == "{g}"', 'Person': f'Person in {subjects}'}
            _, y, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=['activity', 'frequency', 'gender', 'height', 'weight', 'person'])
            self.assertIsInstance(label_map['gender'], dict) 
            self.assertTrue(np.all(y[:, 2] == label_map['gender'][g]))

        h = 170
        with self.subTest(f'filtered by height (height <= {h})'):
            queries = {'Height': f'Height <= {h}', 'Person': f'Person in {subjects}'}
            _, y, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=['activity', 'frequency', 'gender', 'height', 'weight', 'person'])
            self.assertDictEqual(label_map['height'], {})
            self.assertTrue(np.all(y[:, 3] <= h))

        w = 80
        with self.subTest(f'filtered by weight (weight >= {w})'):
            queries = {'Weight': f'Weight >= {w}', 'Person': f'Person in {subjects}'}
            _, y, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=['activity', 'frequency', 'gender', 'height', 'weight', 'person'])
            self.assertDictEqual(label_map['height'], {})
            self.assertTrue(np.all(y[:, 4] >= w))

        freq = 50
        with self.subTest(f'filtered by frequency (frequency == {freq})'):
            queries = {'Frequency': f'Frequency == {freq}', 'Person': f'Person in {subjects}'}
            _, y, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=['activity', 'frequency', 'gender', 'height', 'weight', 'person'])
            self.assertDictEqual(label_map['height'], {})
            self.assertTrue(np.all(y[:, 1] == freq))

        with self.subTest('filtered by subject'):
            queries = {'Person': f'Person in {subjects}'}
            _, y, label_map = self.loader.load(window_size=256, stride=256, ftrim=5, btrim=5, queries=queries, y_labels=['activity', 'frequency', 'gender', 'height', 'weight', 'person'])
            self.assertIsInstance(label_map, dict)
            self.assertSetEqual(set(label_map['person'].keys()), set(subjects))
            self.assertEqual(y[:, 5].min(), 0)
            self.assertEqual(y[:, 5].max(), len(subjects)-1)

        # self.assertIsInstance(raw, tuple)
        # self.assertEqual(len(raw), 2)
        # data, meta = raw
        # self.assertIsInstance(data, list)
        # self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))
        # self.assertIsInstance(meta, pd.DataFrame)
        # self.assertEqual(len(data), len(meta))
        

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        sys.stderr.write('Usage: {} <dataset path> <cache path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])
    cache_path = Path(args[2])

    HASCTest.path = ds_path
    HASCTest.cache_path = cache_path

    unittest.main(verbosity=2, argv=args[0:1])
