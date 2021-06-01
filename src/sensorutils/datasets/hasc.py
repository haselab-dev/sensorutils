"""
hasc challenge

hasc 読み込み関数
"""

import functools
from pathlib import Path
import typing

import numpy as np
import pandas as pd
import pickle
from ..core import split_using_target, split_using_sliding_window

from .base import BaseDataset


__all__ = ['HASC', 'load_meta', 'load']


class HASC(BaseDataset):
    """HASC

    HASCデータセット(HASC-PAC2016)の行動分類を行うためのローダクラス
    """

    supported_queries = [
        'Frequency', 'Gender', 'Height', 'Weight', 'Person'
    ]

    supported_activity_labels = [
        'stay', 'walk', 'jog', 'skip', 'stup', 'stdown',
    ]

    supported_target_labels = ['activity', 'person']

    def __init__(self, path:Path, cache_dir_meta:Path=None):
        super().__init__(path)
        self.cache_dir_meta = cache_dir_meta

        if cache_dir_meta is not None:
            if cache_dir_meta.exists():
                self.meta = pd.read_csv(str(cache_dir_meta), index_col=0)
            else:
                self.meta = load_meta(path)
        else:
            self.meta = load_meta(path)
        
        self.label_map = {'activity': None, 'subject': None}
    
    @classmethod
    def load_from_cache(cls, cache_path:Path):
        with cache_path.open('rb') as fp:
            x, y = pickle.load(fp)
        return x, y
    
    def _filter_with_meta(self, queries):
        """
        Supported Queries
        - Frequency
        - Gender
        - Weight
        - Person 
        """
        if not (set(queries.keys()) < set(self.supported_queries)):
            raise ValueError(f'Unknown queries detected. (Supported: {self.supported_queries})')

        # クエリの整形
        # 括弧は特殊文字として扱われるため無視したい場合はバッククォートで囲む
        if 'Height' in queries.keys():
            queries['Height'] = queries['Height'].replace('Height', '`Height(cm)`')
        if 'Weight' in queries.keys():
            queries['Weight'] = queries['Weight'].replace('Weight', '`Weight(kg)`')
        if 'Person' in queries.keys():
            queries['Person'] = queries['Person'].replace('Person', 'person')
        for k in queries.keys():
            queries[k] = f'({queries[k]})'
        
        query_string = ' & '.join([queries[k] for k in queries.keys()])
        filed_meta = self.meta.query(query_string)
        return filed_meta
    
    def __extract_targets(self, y_labels:list, meta_row:pd.DataFrame) -> np.ndarray:
        targets = []
        for i, yl in enumerate(y_labels):
            t = meta_row._asdict()[yl]
            if t not in self.maps[i]:
                self.maps[i][t] = self.counters[i]
                self.counters[i] += 1
            targets += [self.maps[i][t]]

        return np.array(targets)

    # フィルタリング周りの実装は暫定的
    def load(self, window_size:int, stride:int, ftrim:int=0, btrim:int=0, queries:dict=None, y_labels:typing.Union[str, list]='activity'):
        """HASCデータの読み込みとsliding-window

        Parameters
        ----------
        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        ftrim: int
            セグメント先頭のトリミングサイズ

        btrim: int
            セグメント末尾のトリミングサイズ
        
        queries: dict
            メタ情報に基づいてフィルタリングを行うためのクエリ。
            Keyはフィルタリングのラベル(Supported: Frequency, Height, Weight)
            Valueはクエリ文字列(DataFrame.queryに準拠)
            e.g.
            # サンプリングレートが100Hz and 身長が170cmより大きい and 体重が100kg以上でフィルタリング
            queries = {
                'Frequency': 'Frequency == 100', # サンプリングレートが100Hzのデータのみを取得
                'Height': 'Height > 170',   # 身長が170cmより大きい人
                'Weight': 'Weight >= 100',   # 体重が100kg以上の人
            }
        
        y_labels: Union[str, list]

            ターゲットデータとしてロードするデータの種類を指定．
            サポートする種類は以下の通り(今後拡張予定)．

            - 'activity'
            - 'person'

        Returns
        -------
        (x_frames, y_frames): tuple
            sliding-windowで切り出した入力とターゲットのフレームリスト
        """

        if isinstance(y_labels, str):
            y_labels = [y_labels]
        if not (set(y_labels) <= set(self.supported_target_labels)):
            raise ValueError('include not supported target labels: {}'.format(y_labels))
        target_labels = list(map(lambda x: x.replace('activity', 'act'), y_labels))

        if queries is None:
            filed_meta = self.meta
        else:
            filed_meta = self._filter_with_meta(queries)
        
        segments = load(self.path, filed_meta)
        x_frames = []
        y_frames = []
        self.maps = [{} for _ in y_labels]
        self.counters = [0 for _ in y_labels]
        for meta_row, seg in zip(filed_meta.itertuples(), segments):
            act = meta_row.act
            if act == '0_sequence':
                continue
            ys = self.__extract_targets(target_labels, meta_row)

            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=ftrim, btrim=btrim,
                return_error_value=None)
            if fs is not None:
                x_frames += [fs]
                y_frames += [np.expand_dims(ys, axis=0).repeat(len(fs), axis=0)]
        x_frames = np.concatenate(x_frames)
        y_frames = np.concatenate(y_frames)
        assert len(x_frames) == len(y_frames), 'Mismatch length of x_frames and y_frames'

        self.label_map = dict(zip(y_labels, self.maps))

        return x_frames, y_frames, self.label_map


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