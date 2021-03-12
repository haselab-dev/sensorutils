import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional

__all__ = ['UCIHAR', 'load_meta', 'load']


class UCIHAR(object):
    def __init__(self, ucihar_dir):
        if type(ucihar_dir) == str: ucihar_dir = Path(ucihar_dir)
        self.ucihar_dir = ucihar_dir
        self.load_meta()
    
    def load_meta(self):
        meta = load_meta(self.ucihar_dir)
        self.train_metas = meta['train']
        self.test_metas = meta['test']
   
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
        labels:
            activity labels
        person_id_list:
            the list of person id
        """

        if include_gravity:
            if train:
                sdata = load(self.ucihar_dir, train=True, include_gravity=True)
                metas = self.train_metas
            else:
                sdata = load(self.ucihar_dir, train=False, include_gravity=True)
                metas = self.test_metas
        else:
            if train:
                sdata = load(self.ucihar_dir, train=True, include_gravity=False)
                metas = self.train_metas
            else:
                sdata = load(self.ucihar_dir, train=False, include_gravity=False)
                metas = self.test_metas
 
        flags = np.zeros((sdata.shape[0],), dtype=np.bool)
        if person_list is None: person_list = np.array(PERSONS)
        for person_id in person_list:
            flags = np.logical_or(flags, np.array(metas['person_id'] == person_id))

        sdata = sdata[flags]
        labels = metas['activity'].to_numpy()[flags]
        labels -= 1 # scale: [1, 6] => scale: [0, 5]
        person_id_list = np.array(metas.iloc[flags]['person_id'])

        return sdata, labels, person_id_list

def load_meta(path:Path) -> dict:
    """UCIHAR の meta ファイルを読み込む

    Parameters
    ----------
    path: Path
        UCIHAR ファイルのパス。trainやtestディレクトリがあるパスを指定する

    Returns
    -------
    dict:
        trainとtestのmeta情報
    """
    # train
    train_labels = pd.read_csv(str(path/'train'/'y_train.txt'), header=None)
    train_subjects = pd.read_csv(str(path/'train'/'subject_train.txt'), header=None)
    train_metas = pd.concat([train_labels, train_subjects], axis=1)
    train_metas.columns = ['activity', 'person_id']

    # test
    test_labels = pd.read_csv(str(path/'test'/'y_test.txt'), header=None)
    test_subjects = pd.read_csv(str(path/'test'/'subject_test.txt'), header=None)
    test_metas = pd.concat([test_labels, test_subjects], axis=1)
    test_metas.columns = ['activity', 'person_id']

    return {'train': train_metas, 'test': test_metas}

def load(path:Path, train=True, include_gravity=False):
    """UCIHAR の センサデータを読み込む

    Parameters
    ----------
    path: Path
        UCIHAR ファイルのパス。trainやtestディレクトリがあるパスを指定する
    
    train: bool
        train == True then load train data
        train == False then load test data
    
    include_gravity: bool
        姿勢情報(第0周波数成分)を含むかのフラグ

    Returns
    -------
    """

    if include_gravity:
        if train:
            x = pd.read_csv(str(path/'train'/'Inertial Signals'/'total_acc_x_train.txt'), sep='\s+', header=None).to_numpy()
            y = pd.read_csv(str(path/'train'/'Inertial Signals'/'total_acc_y_train.txt'), sep='\s+', header=None).to_numpy()
            z = pd.read_csv(str(path/'train'/'Inertial Signals'/'total_acc_z_train.txt'), sep='\s+', header=None).to_numpy()
        else:
            x = pd.read_csv(str(path/'test'/'Inertial Signals'/'total_acc_x_test.txt'), sep='\s+', header=None).to_numpy()
            y = pd.read_csv(str(path/'test'/'Inertial Signals'/'total_acc_y_test.txt'), sep='\s+', header=None).to_numpy()
            z = pd.read_csv(str(path/'test'/'Inertial Signals'/'total_acc_z_test.txt'), sep='\s+', header=None).to_numpy()
    else:
        if train:
            x = pd.read_csv(str(path/'train'/'Inertial Signals'/'body_acc_x_train.txt'), sep='\s+', header=None).to_numpy()
            y = pd.read_csv(str(path/'train'/'Inertial Signals'/'body_acc_y_train.txt'), sep='\s+', header=None).to_numpy()
            z = pd.read_csv(str(path/'train'/'Inertial Signals'/'body_acc_z_train.txt'), sep='\s+', header=None).to_numpy()
        else:
            x = pd.read_csv(str(path/'test'/'Inertial Signals'/'body_acc_x_test.txt'), sep='\s+', header=None).to_numpy()
            y = pd.read_csv(str(path/'test'/'Inertial Signals'/'body_acc_y_test.txt'), sep='\s+', header=None).to_numpy()
            z = pd.read_csv(str(path/'test'/'Inertial Signals'/'body_acc_z_test.txt'), sep='\s+', header=None).to_numpy()

    sensor_data = np.concatenate([x[:, np.newaxis, :], y[:, np.newaxis, :], z[:, np.newaxis, :]], axis=1)

    return sensor_data

 
PERSONS = list(range(1, 31))
ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
 
