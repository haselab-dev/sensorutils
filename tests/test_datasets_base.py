import sys
from typing import Type
import unittest

from pathlib import Path

sys.path.append('../src/')
from sensorutils.datasets.base import check_path


class BaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test__check_path(self):
        # unexpected type
        with self.assertRaises(TypeError):
            check_path(0)
        with self.assertRaises(TypeError):
            check_path([0])

        # not exist path
        path = Path('hogepiyofuga')
        while path.exists():
            path = Path(str(path) + '_')
        with self.assertRaises(FileNotFoundError):
            check_path(path)

if __name__ == '__main__':
    unittest.main(verbosity=2)
