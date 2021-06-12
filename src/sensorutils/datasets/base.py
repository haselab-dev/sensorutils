"""
データセットローダの抽象クラス
"""

from pathlib import Path

class BaseDataset(object):
    def __init__(self, path:Path):
        self.path = path
    
    def load(self, *args):
        raise NotImplementedError
