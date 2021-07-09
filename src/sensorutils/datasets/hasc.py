"""HASC dataset

http://hasc.jp/
"""

import functools
from pathlib import Path
from typing import List, Tuple, Union, Optional, Iterable

import numpy as np
import pandas as pd
import pickle
from ..core import split_using_sliding_window

from .base import BaseDataset, check_path


__all__ = ['HASC', 'load', 'load_raw', 'load_meta']


class HASC(BaseDataset):
    """
    HASCデータセット(HASC-PAC2016)に記録されているセンサデータとメタデータを読み込む．

    Parameters
    ----------
    path: Path
        HASCデータセットのパス．BasicActivityディレクトリの親ディレクトリのパスを指定する．
    
    meta_cache_path: Optional[Path]
        メタデータのキャッシュファイルのパス．
        何も指定されない場合(meta_cache_path=None)，メタデータの作成を行うが，キャッシュファイルは作成しない．
        ファイル名が指定された場合，そのファイルが存在すればそこからメタデータを読み込み，存在しなければメタデータの作成を行い指定したファイルパスにダンプする．

    See Also
    --------
    メタデータの読み込みは非常に時間がかかるため，キャッシュファイルを活用することをおすすめする．
    """

    supported_queries = [
        'Frequency', 'Gender', 'Height', 'Weight', 'Person'
    ]

    supported_activity_labels = [
        'stay', 'walk', 'jog', 'skip', 'stup', 'stdown',
    ]

    supported_target_labels = ['activity', 'frequency', 'gender', 'height', 'weight', 'person']

    def __init__(self, path:Path, meta_cache_path:Optional[Path]=None):
        super().__init__(path)
        self.cache_dir_meta = meta_cache_path

        if meta_cache_path is not None:
            if meta_cache_path.exists():
                self.meta = pd.read_csv(str(meta_cache_path), index_col=0)
            else:
                self.meta = load_meta(path)
                self.meta.to_csv(str(meta_cache_path))
        else:
            self.meta = load_meta(path)

        dtypes = {
            'LogVersion': np.float64,
            'Frequency': np.float64,
            'Height(cm)': np.float64,
            'Weight(kg)': np.float64,
            'Pace(cm)': np.float64,
            'HeightOfOneStairStep(cm)': np.float64,
            'UseHistory': np.float64,
            'Count': np.float64,
            # '﻿LogVersion': np.float64,
        }
        self.meta = self.meta.replace('', np.nan)
        self.meta = self.meta.astype(dtypes)
        
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
    
    def __extract_targets(self, y_labels:Iterable, meta_row:pd.Series) -> np.ndarray:
        if not isinstance(meta_row, pd.Series):
            raise TypeError('meta_row must be "pd.Series", but {}'.format(type(meta_row)))
        targets = []
        for i, yl in enumerate(y_labels):
            # metaデータのカラム名に変換
            ylbl2col = {
                'activity': 'act',
                'frequency': 'Frequency',
                'gender': 'Gender',
                'height': 'Height(cm)',
                'weight': 'Weight(kg)',
                'person': 'person',
            }
            assert set(ylbl2col.keys()) == set(self.supported_target_labels), 'table for converting "y_label" to column is not enough'

            # t = meta_row._asdict()[yl]
            t = meta_row.to_dict()[ylbl2col[yl]]
            cat_labels = ('activity', 'gender', 'person')
            if yl in cat_labels:
                if t not in self.maps[i]:
                    # self.maps: List[dict], self.counters: List[int]
                    self.maps[i][t] = self.counters[i]
                    self.counters[i] += 1
                targets += [self.maps[i][t]]
            elif yl in (set(self.supported_target_labels) - set(cat_labels)):
                targets += [t]
            # else:
            #     # not reach

        return np.array(targets)

    # フィルタリング周りの実装は暫定的
    def load(self, window_size:int, stride:int, ftrim:int=0, btrim:int=0, queries:Optional[dict]=None, y_labels:Union[str, list]='activity') -> Tuple[np.ndarray, np.ndarray]:
        """
        HASCデータセットを読み込み，sliding-window処理を行ったデータを返す．
        ここでは3軸加速度センサデータのみを扱う．

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
            メタ情報に基づいてフィルタリングを行うためのクエリ．

            Keyはフィルタリングのラベル(Supported: Frequency, Height, Weight, Gender)

            Valueはクエリ文字列(DataFrame.queryに準拠)

            詳しい使い方は後述．
        
        y_labels: Union[str, list]

            ターゲットデータとしてロードするデータの種類を指定．
            listで指定した場合，その順序が返されるターゲットラベルにも反映される．
            サポートする種類は以下の通り(今後拡張予定)．

            - 'activity'
            - 'frequency'
            - 'gender'
            - 'height'
            - 'weight'
            - 'person'

        Returns
        -------
        (x_frames, y_frames): Tuple[np.ndarray, np.ndarray]
            sliding-windowで切り出した入力とターゲットのフレームリスト．

            x_framesは3次元配列で構造は大まかに(Batch, Channels, Frame)のようになっている．
            Channelsは加速度センサの軸を表しており，先頭からx, y, zである．

            y_framesはy_labelsで指定したターゲットラベルであり，
            y_frames(axis=1)のラベルの順序はy_labelsのものが保持されている．
        
        Examples
        --------
        サンプリングレートが100Hz and 身長が170cmより大きい and 体重が100kg以上でフィルタリング
        >>> hasc_path = Path('/path/to/dataset/HASC-PAC2016/')  # HASCデータセットパス
        >>> hasc = HASC(hasc_path, Path('D:/datasets/HASC-PAC2016/BasicActivity/hasc.csv'))
        >>> queries = {
        >>>     'Frequency': 'Frequency == 100', # サンプリングレートが100Hzのデータのみを取得
        >>>     'Height': 'Height > 170',   # 身長が170cmより大きい人
        >>>     'Weight': 'Weight >= 100',   # 体重が100kg以上の人
        >>> }
        >>>
        >>> y_labels = ['activity', 'person']    # ターゲットラベルとしてacitivityとpersonを取り出す
        >>>
        >>> # yのaxis=1にはactivity, personの順でターゲットラベルが格納されている．
        >>> x, y, act2id = hasc.load(window_size=256, stride=256, queries=queries, ftrim=2*100, btrim=2*100, y_labels=y_labels)
        >>>
        >>> print(f'x: {x.shape}, y: {y.shape}')
        >>> # > x: (?, 3, 256), y: (?, 2)
        """

        if isinstance(y_labels, str):
            y_labels = [y_labels]
        if not (set(y_labels) <= set(self.supported_target_labels)):
            raise ValueError('include not supported target labels: {}'.format(y_labels))

        if queries is None:
            filed_meta = self.meta
        else:
            filed_meta = self._filter_with_meta(queries)
        
        segments, _ = load(self.path, meta=filed_meta)
        x_frames = []
        y_frames = []
        self.maps = [{} for _ in y_labels]
        self.counters = [0 for _ in y_labels]
        for (_, meta_row), seg in zip(filed_meta.iterrows(), segments):
            act = meta_row['act']
            if act == '0_sequence':
                continue
            ys = self.__extract_targets(tuple(y_labels), meta_row)

            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=ftrim, btrim=btrim,
                return_error_value=None)
            if fs is not None:
                x_frames += [fs]
                y_frames += [np.expand_dims(ys, axis=0).repeat(len(fs), axis=0)]
        x_frames = np.concatenate(x_frames)
        x_frames = x_frames.transpose(0, 2, 1)
        y_frames = np.concatenate(y_frames)
        assert len(x_frames) == len(y_frames), 'Mismatch length of x_frames and y_frames'

        self.label_map = dict(zip(y_labels, self.maps))

        return x_frames, y_frames, self.label_map


def load(path:Union[Path,str], meta:pd.DataFrame) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Function for loading HASC dataset

    Parameters
    ----------
    path: Union[Path, str]
        Directory path of HASC dataset, which is parent directory of "BasicActivity" directory.
    
    meta: pd.DataFrame
        meta data loaded by 'load_meta'.
    
    Returns
    -------
    data, meta: List[pd.DataFrame], pd.DataFrame
        Sensor data segmented by activity and subject.
    
    See Also
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta.iloc[0] is meta data of data[0].
    """
    path = check_path(path)

    raw = load_raw(path, meta)
    data, meta = reformat(raw)
    return data, meta


def load_meta(path:Path) -> pd.DataFrame:
    """Function for loading meta data of HASC dataset

    Parameters
    ----------
    path: Path
        Directory path of HASC dataset, which is parent directory of "BasicActivity" directory.

    Returns
    -------
    metas: pd.DataFrame:
        meta data of HASC dataset.
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


def load_raw(path:Path, meta:Optional[pd.DataFrame]=None) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Function for loading raw data of HASC dataset

    Parameters
    ----------
    path: Path
        Directory path of HASC dataset, which is parent directory of "BasicActivity" directory.
    
    meta: pd.DataFrame
        meta data loaded by 'load_meta'.
    
    Returns
    -------
    data, meta: List[pd.DataFrame], pd.DataFrame
        raw data of HASC dataset.

        Each item in 'data' is a part of dataset, which is splited by subject.
    
    See Also
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta.iloc[0] is meta data of data[0].
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
    
    if meta is None:
        meta = load_meta(path)

    path = path / 'BasicActivity'
    data = [
        read_acc(path / row.act / row.person / '{}-acc.csv'.format(row.file))
        for row in meta.itertuples()
    ]

    return data, meta


def reformat(raw) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Function for reformating

    Parameters
    ----------
    raw:
        data loaded by 'load_raw'
    
    Returns
    -------
    data, meta: List[pd.DataFrame], pd.DataFrame
        Sensor data segmented by activity and subject.

    See Also
    --------
    The order of 'data' and 'meta' correspond.

    e.g. meta.iloc[0] is meta data of data[0].
    """
    data, meta = raw
    return data, meta

