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
from ..core import split_by_sliding_window

class HASC:
    """HASC

    HASCデータセット(HASC-PAC2016)の行動分類を行うためのローダクラス
    """

    supported_queries = [
        'Frequency', 'Gender', 'Height', 'Weight',
    ]

    supported_activity_labels = [
        'stay', 'walk', 'jog', 'skip', 'stup', 'stdown',
    ]

    def __init__(self, path:Path, cache_dir_meta:Path=None):
        self.path = path
        self.cache_dir_meta = cache_dir_meta

        if cache_dir_meta.exists():
            self.meta = pd.read_csv(str(cache_dir_meta), index_col=0)
        else:
            self.meta = load_meta(path)
    
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
        """
        if not (set(queries.keys()) < set(self.supported_queries)):
            raise ValueError(f'Unknown queries detected. (Supported: {self.supported_queries})')

        # クエリの整形
        # 括弧は特殊文字として扱われるため無視したい場合はバッククォートで囲む
        if 'Height' in queries.keys():
            queries['Height'] = queries['Height'].replace('Height', '`Height(cm)`')
        if 'Weight' in queries.keys():
            queries['Weight'] = queries['Weight'].replace('Weight', '`Weight(kg)`')
        
        query_string = ' & '.join([queries[k] for k in queries.keys()])
        filed_meta = self.meta.query(query_string)
        return filed_meta

    # フィルタリング周りの実装は暫定的
    def load(self, window_size:int, stride:int, ftrim:int=0, btrim:int=0, queries:dict=None):
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

        Returns
        -------
        (x_frames, y_frames): tuple
            sliding-windowで切り出した入力とターゲットのフレームリスト
        """

        if queries is None:
            filed_meta = self.meta
        else:
            filed_meta = self._filter_with_meta(queries)
        
        segments = load(self.path, filed_meta)
        x_frames = []
        y_frames = []
        act2id = {}
        act_id_counter = 0
        for meta_row, seg in zip(filed_meta.itertuples(), segments):
            act = meta_row.act
            if act == '0_sequence':
                continue
            if act not in act2id.keys():
                act2id[act] = act_id_counter
                act_id_counter += 1

            fs = split_by_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=ftrim, btrim=btrim,
                return_error_value=None)
            if fs is not None:
                x_frames += [fs]
                y_frames += [np.array([act2id[act]]).repeat(len(fs))]
        x_frames = np.concatenate(x_frames)
        y_frames = np.concatenate(y_frames)
        assert len(x_frames) == len(y_frames), 'Mismatch length of x_frames and y_frames'
        return x_frames, y_frames, act2id


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