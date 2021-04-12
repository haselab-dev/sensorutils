"""
データセットローダの抽象クラス
"""

import typing
from pathlib import Path

class BaseDataset(object):
    def __init__(self, path:Path):
        self.path = path
    
    def act2id(self):
        raise NotImplementedError

    def subject2id(self):
        raise NotImplementedError
    
    @property
    def activity_label_to_id(self):
        return self.act2id()
    
    @property
    def subject_label_to_id(self):
        return self.subject2id()
    
    def load(self, *args):
        raise NotImplementedError
