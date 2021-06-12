import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union, Optional

from .base import BaseDataset


__all__ = ['UCIHAR', 'load', 'load_raw', 'load_meta']


# Meta Info
PERSONS = list(range(1, 31))
ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']


class UCIHAR(BaseDataset):
    def __init__(self, path:Union[str, Path]):
        if type(path) == str: path = Path(path)
        super().__init__(path)
        # self.load_meta()
    
    # 廃止予定
    def load_meta(self):
        # meta = load_meta(self.path)
        # self.train_metas = meta['train']
        # self.test_metas = meta['test']
        self.metas = load_meta(self.path)
   
    def load(self, train:bool=True, person_list:Optional[list]=None, include_gravity:bool=True) -> tuple:
        """Sliding-Windowをロード

        Parameters
        ----------
        train: bool
            select train data or test data. if True then return train data.
        person_list: Option[list]
            specify persons.
        include_gravity: bool
            select whether or not include gravity information.
       
        Returns
        -------
        sensor_data:
            sliding-windows
        targets:
            activity and subject labels
            subjectラベルはデータセット内の値をそのまま返すため，分類等で用いる際はラベルの再割り当てが必要となることに注意
        
        See Also
        --------
        Range of activity label: [0, 5]
        Range of subject label :
            if train is True: [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30] (21 subjects)
            else : [2, 4, 9, 10, 12, 13, 18, 20, 24] (9 subjects)
        """

        if include_gravity:
            sdata, metas = load(self.path, include_gravity=True)
        else:
            sdata, metas = load(self.path, include_gravity=False)
 
        sdata = np.stack(sdata).transpose(0, 2, 1)
        flags = np.zeros((sdata.shape[0],), dtype=bool)
        if person_list is None: person_list = np.array(PERSONS)
        for person_id in person_list:
            flags = np.logical_or(flags, np.array(metas['person_id'] == person_id))

        sdata = sdata[flags]
        labels = metas['activity'].to_numpy()[flags]
        labels -= 1 # scale: [1, 6] => scale: [0, 5]
        person_id_list = np.array(metas.iloc[flags]['person_id'])
        train_flags = np.array(metas['train'].iloc[flags], dtype=np.int8)
        targets = np.stack([labels, person_id_list, train_flags]).T

        if train:
            l = 1
        else:
            l = 0
        sdata = sdata[targets[:, 2] == l]
        targets = targets[targets[:, 2] == l]

        return sdata, targets


def load(path:Path, include_gravity:bool) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    raw = load_raw(path, include_gravity=include_gravity)
    data, meta = reformat(raw)
    # assert isinstance(data, list) and all(isinstance(d, pd.DataFrame) for d in data), '[debug] different type on "data", data: {}[{}]'.format(type(data), type(data[0]))
    # assert isinstance(meta, pd.DataFrame), '[debug] different type on "meta", meta: {}'.format(type(meta))
    # assert len(data) == len(meta), '[debug] different shape, data: {}, meta: {}'.format(len(data), meta.shape)
    return data, meta


def load_meta(path:Path) -> pd.DataFrame:
    """UCIHAR の meta ファイルを読み込む

    Parameters
    ----------
    path: Path
        UCIHAR ファイルのパス。trainやtestディレクトリがあるパスを指定する

    Returns
    -------
    metas: pd.DataFrame
        trainとtestのmeta情報
    """
    # train
    train_labels = pd.read_csv(str(path/'train'/'y_train.txt'), header=None)
    train_subjects = pd.read_csv(str(path/'train'/'subject_train.txt'), header=None)
    train_metas = pd.concat([train_labels, train_subjects], axis=1)
    train_metas.columns = ['activity', 'person_id']
    train_metas['train'] = True

    # test
    test_labels = pd.read_csv(str(path/'test'/'y_test.txt'), header=None)
    test_subjects = pd.read_csv(str(path/'test'/'subject_test.txt'), header=None)
    test_metas = pd.concat([test_labels, test_subjects], axis=1)
    test_metas.columns = ['activity', 'person_id']
    test_metas['train'] = False

    metas = pd.concat([train_metas, test_metas], axis=0)
    dtypes = {'activity': np.int8, 'person_id': np.int8, 'train': bool}
    metas = metas.astype(dtypes)

    return metas


def load_raw(path:Path, include_gravity:bool) -> Tuple[np.ndarray, pd.DataFrame]:
    """UCIHAR の センサデータを読み込む

    Parameters
    ----------
    path: Path
        UCIHAR ファイルのパス。trainやtestディレクトリがあるパスを指定する
    
    include_gravity: bool
        姿勢情報(第0周波数成分)を含むかのフラグ

    Returns
    -------
    sensor_data, meta: Tuple[np.ndarray, pd.DataFrame]
        Shape of sensor_data is (?, 3, 128).

    """

    if include_gravity:
        x_tr = pd.read_csv(str(path/'train'/'Inertial Signals'/'total_acc_x_train.txt'), sep=r'\s+', header=None).to_numpy()
        y_tr = pd.read_csv(str(path/'train'/'Inertial Signals'/'total_acc_y_train.txt'), sep=r'\s+', header=None).to_numpy()
        z_tr = pd.read_csv(str(path/'train'/'Inertial Signals'/'total_acc_z_train.txt'), sep=r'\s+', header=None).to_numpy()
        x_ts = pd.read_csv(str(path/'test'/'Inertial Signals'/'total_acc_x_test.txt'), sep=r'\s+', header=None).to_numpy()
        y_ts = pd.read_csv(str(path/'test'/'Inertial Signals'/'total_acc_y_test.txt'), sep=r'\s+', header=None).to_numpy()
        z_ts = pd.read_csv(str(path/'test'/'Inertial Signals'/'total_acc_z_test.txt'), sep=r'\s+', header=None).to_numpy()
    else:
        x_tr = pd.read_csv(str(path/'train'/'Inertial Signals'/'body_acc_x_train.txt'), sep=r'\s+', header=None).to_numpy()
        y_tr = pd.read_csv(str(path/'train'/'Inertial Signals'/'body_acc_y_train.txt'), sep=r'\s+', header=None).to_numpy()
        z_tr = pd.read_csv(str(path/'train'/'Inertial Signals'/'body_acc_z_train.txt'), sep=r'\s+', header=None).to_numpy()
        x_ts = pd.read_csv(str(path/'test'/'Inertial Signals'/'body_acc_x_test.txt'), sep=r'\s+', header=None).to_numpy()
        y_ts = pd.read_csv(str(path/'test'/'Inertial Signals'/'body_acc_y_test.txt'), sep=r'\s+', header=None).to_numpy()
        z_ts = pd.read_csv(str(path/'test'/'Inertial Signals'/'body_acc_z_test.txt'), sep=r'\s+', header=None).to_numpy()

    x = np.concatenate([x_tr, x_ts], axis=0)
    y = np.concatenate([y_tr, y_ts], axis=0)
    z = np.concatenate([z_tr, z_ts], axis=0)

    sensor_data = np.concatenate([x[:, np.newaxis, :], y[:, np.newaxis, :], z[:, np.newaxis, :]], axis=1)
    sensor_data = sensor_data.astype(np.float64)
    meta = load_meta(path)

    return sensor_data, meta

 
def reformat(raw) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    data, meta = raw
    # assert len(data) == len(meta), 'data and meta are not same length ({}, {})'.format(len(data), len(meta))
    data = list(map(lambda x: pd.DataFrame(x.T, columns=['x', 'y', 'z']), data))
    return data, meta

