"""
データセットローダの抽象クラス

新しいデータセットのローダクラスを実装する場合は必ずこのクラスを継承する．
"""

from pathlib import Path

class BaseDataset(object):
    def __init__(self, path:Path):
        self.path = path
    
    def load(self, *args):
        raise NotImplementedError
