"""
hasc challenge

hasc 読み込み関数
"""

import functools
from pathlib import Path
import typing

import pandas as pd


def load_meta(path:Path) -> pd.DataFrame:
    """HASC の meta ファイルを読み込む。

    Parameters
    ----------
    path: Path
        HASC ファイルのパス。BasicActivity のあるディレクトリを指定すること。

    Returns
    -------
    pd.DataFrame:
        meta ファイルの DataFrame。
    """
    def replace_meta_str(s:str) -> str:
        s = s.replace('：', ':')
        s = s.replace('TerminalPosition:TerminalPosition:',
                            'TerminalPosition:')
        s = s.replace('sneakers:leathershoes', 'sneakers;leathershoes')
        s = s.replace('Frequency(Hz)', 'Frequency')
        return s

    def read_meta(file:Path):
        with file.open(mode='rU', encoding='utf-8') as f:
            ret = [s.strip() for s in f.readlines()]
            ret = filter(bool, ret)
            ret = [replace_meta_str(s) for s in ret]
            ret = [e.partition(':')[0::2] for e in ret]
            ret = {key.strip(): val.strip() for key, val in ret}
        act, person, file_name = file.parts[-3:]
        ret['act'] = act
        ret['person'] = person
        ret['file'] = file_name.split('.')[0]
        return ret

    path = path / 'BasicActivity'
    assert path.exists(), '{} is not exists.'.format(str(path))
    files = path.glob('**/**/*.meta')
    metas = [read_meta(p) for p in files]
    metas = functools.reduce(
        lambda a, b: a.append([b]),
        metas,
        pd.DataFrame()
    )
    return metas


def load(path:Path, meta:pd.DataFrame) -> typing.List[pd.DataFrame]:
    """HASC の行動加速度センサデータの読み込み関数。
    予め meta を読み込む必要がある。

    pd.DataFrame の itertuple が順序を守っていたはずなので、
    返すデータのリストの順番は meta ファイルと一致する。

    Parameters
    ----------
    path: Path
        BasicActivity がある HASC のファイルパス

    meta: pd.DataFrame
        load_meta 関数で返された meta ファイル

    Returns
    -------
    List[pd.DataFrame]:
        行動加速度センサデータのリスト。
    """
    def read_acc(path:Path) -> pd.DataFrame:
        if path.exists():
            try:
                ret = pd.read_csv(str(path), index_col=0, names=('x', 'y', 'z'))
                return ret
            except:
                print('[load] different data format:', str(path))
        else:
            print('[load] not found:', str(path))
        return pd.DataFrame()

    path = path / 'BasicActivity'
    data = [
        read_acc(path / row.act / row.person / '{}-acc.csv'.format(row.file))
        for row in meta.itertuples()
    ]
    return data