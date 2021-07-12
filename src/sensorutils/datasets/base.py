"""
データセットローダの抽象クラス

新しいデータセットのローダクラスを実装する場合は必ずこのクラスを継承する．
"""

import os, errno

from pathlib import Path
from typing import Union

class BaseDataset(object):
    def __init__(self, path:Path):
        self.path = path
    
    def load(self, *args):
        raise NotImplementedError

def check_path(path:Union[Path,str]) -> Path:
    """
    Check a path

    Parameters
    ----------
    path: Union[Path, str]
        any path
    
    Returns
    -------
    path: Path
        checked path
    
    Raises
    ------
    TypeError
        * if type of path is not Path or str

    FileNotFoundError
        * if path does not exist
    """

    # type check
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise TypeError('expected type of "path" is Path or str, but {}'.format(type(path)))
    # else:
    #     # type(path) == Path
    
    # check path existence
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    
    return path
