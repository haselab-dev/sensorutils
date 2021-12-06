import sys
import unittest

import numpy as np
import pandas as pd
import itertools
from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.dsads import load, load_raw, reformat


class DSADSTest(unittest.TestCase):
    path = None
    activities = list(range(19))
    subjects = list(range(1, 9))

    @classmethod
    def setUpClass(cls) -> None:
        if cls.path is None:
            raise RuntimeError('dataset path is not specified')

    def setUp(self):
        # self.loader = DSADS(self.path)
        pass

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
        self.assertTrue(all([list(m.columns) == ['activity', 'subject'] for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['activity'] == np.dtype(np.int8) and \
            m.dtypes['subject'] == np.dtype(np.int8) \
            for m in meta
        ]))

        ## data check
        self.assertTrue(all([
            set(np.unique(m['subject'])) == set([m['subject'].iloc[0]]) for m in meta
        ]))
        self.assertTrue(all([
            set(np.unique(m['activity'])) == set([m['activity'].iloc[0]]) for m in meta
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([
            list(d.columns) == \
            ['T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag', 'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag', 'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag', 'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']
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
            'T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag',
            'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag',
            'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag',
            'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag',
            'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag',
            'activity', 'subject',
        ])
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
        self.assertTrue(all([
            set(np.unique(r['subject'])) == set([r['subject'].iloc[0]]) for r in raw 
        ]))
        self.assertSetEqual(
            set(itertools.chain(*[np.unique(r['activity']).tolist() for r in raw])),
            set(self.activities)    # performed activity(protocol) + others
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
        self.assertTrue(all([list(m.columns) == ['activity', 'subject'] for m in meta]))

        ## data type check
        self.assertTrue(all([
            m.dtypes['activity'] == np.dtype(np.int8) and \
            m.dtypes['subject'] == np.dtype(np.int8) \
            for m in meta
        ]))

        ## data check
        self.assertTrue(all([
            set(np.unique(m['subject'])) == set([m['subject'].iloc[0]]) for m in meta
        ]))
        self.assertTrue(all([
            set(np.unique(m['activity'])) == set([m['activity'].iloc[0]]) for m in meta
        ]))

        # data
        ## type check
        self.assertIsInstance(data, list)
        self.assertTrue(all(isinstance(d, pd.DataFrame) for d in data))

        ## shape and column check
        self.assertTrue(all([
            list(d.columns) == \
            ['T_xacc', 'T_yacc', 'T_zacc', 'T_xgyro', 'T_ygyro', 'T_zgyro', 'T_xmag', 'T_ymag', 'T_zmag', 'RA_xacc', 'RA_yacc', 'RA_zacc', 'RA_xgyro', 'RA_ygyro', 'RA_zgyro', 'RA_xmag', 'RA_ymag', 'RA_zmag', 'LA_xacc', 'LA_yacc', 'LA_zacc', 'LA_xgyro', 'LA_ygyro', 'LA_zgyro', 'LA_xmag', 'LA_ymag', 'LA_zmag', 'RL_xacc', 'RL_yacc', 'RL_zacc', 'RL_xgyro', 'RL_ygyro', 'RL_zgyro', 'RL_xmag', 'RL_ymag', 'RL_zmag', 'LL_xacc', 'LL_yacc', 'LL_zacc', 'LL_xgyro', 'LL_ygyro', 'LL_zgyro', 'LL_xmag', 'LL_ymag', 'LL_zmag']
            for d in data
        ]))

        ## data type check
        self.assertTrue(all([
            d.dtypes[c] == np.dtype(np.float64) \
            for d in data for c in data[0].columns
        ]))


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        sys.stderr.write('Usage: {} <dataset path>'.format(args[0]))
        sys.exit(1)
    
    ds_path = Path(args[1])

    DSADSTest.path = ds_path

    unittest.main(verbosity=2, argv=args[0:1])
