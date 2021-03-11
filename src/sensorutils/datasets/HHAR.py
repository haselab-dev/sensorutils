import numpy as np
import pandas as pd

from pathlib import Path
from ..core import split_using_sliding_window, split_using_target
import typing

class HHAR:
    """HHAR 
    
    HHAR <https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition> データセットの行動分類を行うためのローダクラス

    """

    def __init__(self, path:typing.Union[str,Path], device_type='w'):
        """
        Parameters
        ----------
        dataset_path : Union[str,Path]
            Directory Path that include csv file: Phones_[accelerometer,gyroscope].csv, Watch_[accelerometer,gyroscope].csv
        """

        if type(path) is str:
            path = Path(path)
        self.path = path

        self.device_type = device_type


    def load(self, window_size:int, stride:int):
        """HHARの読み込みとsliding-window

        Parameters
        ----------
        window_size: int
            フレーム分けするサンプルサイズ

        stride: int
            ウィンドウの移動幅

        Returns
        -------
        (x_frames, y_frames): tuple
            sliding-windowで切り出した入力とターゲット(ターゲット以外のメタ情報も含む)のフレームリスト
        """
        segments = load(self.path, device_type=self.device_type)

        frames = []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=0, btrim=0,
                return_error_value=None)
            if fs is not None:
                frames += [fs]
            else:
                print('no frame')
        frames = np.concatenate(frames)
        # print(frames[0, 0, :])
        x_frames = frames[:, :, 3:6]
        y_frames = frames[:, 0, 6:] # 1つのフレームに1つのラベルを含むメタ情報でいいからwindow_size分いらないため0を指定
        return x_frames, y_frames


def load(path:Path, device_type:str='Watch') -> typing.List[pd.DataFrame]:
    """HHAR の加速度センサデータの読み込み関数

    Parameters
    ----------
    path : Path
        [description]
    device_type : str
        [description]

    Returns
    -------
    typing.List[pd.DataFrame]
        [description]
    """  
    # Returns
    # -------
    # segments : Dict[str, List]
    #     key: User Name等の識別子
    #     Value: 連続した同じ行動(可変長)を1つの要素とするリスト
    
    
    if (device_type[0] == 'w') or (device_type[0] == 'W'):
        device_type = DEVICE_TYPE[1]
    else:
        device_type = DEVICE_TYPE[0] # default
    print(device_type)

    # prepare csv path
    csv_files = list(path.glob(device_type + "*.csv"))

    print("CSV files: {}".format(csv_files))
    if len(csv_files) == 0:
        raise FileNotFoundError("Not found csv files in {}".format(str(path)))

    csv_files.sort() # order: Accelerometer, Gyroscope
    # sensor_type_list = ['accelerometer', 'gyro'] # key
    sensor_type_list = ['accelerometer'] # key

    raw_data: typing.Dict[str, pd.DataFrame] = dict()
    for sensor_type, csv_path in zip(sensor_type_list, csv_files):
        df = pd.read_csv(csv_path)
        df['gt'] = df['gt'].fillna('null')
        df['gt'] = df['gt'].map(__act2id)

        raw_data[sensor_type] = df

        print(f"{sensor_type}: {df.shape}")

    
    # load & formatting
    # convert dataframe to segments
    df_acc: pd.DataFrame = raw_data['accelerometer']
    # df_gyro: pd.DataFrame = data['gyro']
    df = df_acc

    # Dict[str, List]
    #     key: User Name等の識別子
    #     Value: 連続した同じ行動(可変長)を1つの要素とするリスト
    segments:typing.Dict[str, typing.List[pd.DataFrame]] = {}

    # PersonごとにDataFrameを分ける
    for person in PERSONS:
        df_person = df[df.User == person]

        # DataFrameをIndexでsegmentsに分ける -> 計測ごとに分かれる(一回の計測に複数の行動が含まれる)
        # 0から始まっていないIndexが存在しているため少々対応が難しい
        # index[1:] - index != 1のインデックスで切ればよい
        # このデータセットでは行動の順番がIndexを以下のように内包していそう．
        # Index         0,1,2,3,4,5,6,7,8,9, 0,1,2,3
        # gt(Activity)  3,3,3,3,3,3,3,3,3,3, 0,0,0,0
        # そのためIndexで分けずに行動のみで分ける

        # 人で分けられたaccelerator gyroをくっつける

        # segmentを行動で分ける
        import itertools
        sub_segs = split_using_target(df_person.values, df_person['gt'].values)
        # sub_segs = list(itertools.chain(*[sub_segs[k] for k in sub_segs.keys()]))  # 連結
        sub_segs = list(itertools.chain(*sub_segs.values()))  # 連結
        sub_segs = list(map(lambda x: pd.DataFrame(x, columns=df_person.columns), sub_segs))

        # TODO もう少しきれいにしたい．人ごとの処理ではなく，他の分割もして．．．
        # for ラベルが連続している箇所が存在してしまっていることへの対処
        # 327  10211  1424689035220     467339764000  4.65014 -0.756568   8.53353    g     gear     gear_2  1
        # 328  10212  1424689035230     467349562000  5.41209 -0.750582   8.61434    g     gear     gear_2  1
        # 329      0  1424689044746  200228355692555 -11.4626  -2.24576  -1.81941    g  lgwatch  lgwatch_1  1 <- これの対処
        loop = enumerate(sub_segs)
        for i, seg in loop:
            seg_labels = seg['Device'].values
            flg = np.all(seg_labels == seg_labels[0])
            if flg == False:
                df_sub = sub_segs.pop(i)
                s = split_using_target(df_sub, __Category2id(df_sub['Device'].values))
                # print(s)
                # print(list(s.keys()))
                # print("------------")
                s = list(itertools.chain(*s.values()))  # 連結
                # print(s)
                sub_segs.extend(s)

        segments[person] = sub_segs

    # print("groupby ------------------------------")
    # # df_gb = df.groupby(by=['User', 'Model', 'Device', 'gt'])
    # # print(df_gb.size())
    # # print(df_gb.groups[('a', 'gear', 'gear_1', 0)])
    # # segments:typing.List[pd.DataFrame] = list(dict(list(df.groupby(by=['User', 'Model', 'Device', 'gt'])).values()))
    # segments:typing.Dict[typing.List] = dict(list(df.groupby(by=['User', 'Model', 'Device', 'gt']))) # これだと間に他の動作が入っていたとしても1つにまとめてしまう
    # print("--------------------------------------")
    # print()

    # print(len(segments))
    # # print(segments['a'])
    # for name, l in segments.items():
    #     print(f"{name}: {len(l)}")
    
    import itertools
    # segments = list(itertools.chain(*[segments[k] for k in segments.keys()]))
    segments = list(itertools.chain(*segments.values()))

    return segments


def __id2act(act_id:int) -> str:
    return ACTIVITIES[act_id]

def __act2id(act:str) -> int:
    ret =  [k for k, v in ACTIVITIES.items() if v == act]
    if len(ret) <= 0:
        return 0
    return ret[0]

# データセットのメタデータ
# ATTRIBUTES = ['acc','gyro']

DEVICE_TYPE = ['Phone', 'Watch']

PERSONS = ['a','b','c','d','e','f','g','h','i']

ACTIVITIES = {
    1: 'bike', 
    2: 'sit', 
    3: 'stand', 
    4: 'walk', 
    5: 'stairsup', 
    6: 'stairsdown',
    0: 'null',
}

# Column = [
#     'Index', 
#     'Arrival_Time', 
#     'Creation_Time', 
#     'x', 
#     'y', 
#     'z', 
#     'User',
#     'Model', 'Device', 'gt',
# ]

def __Category2id(array: np.array):
    mapper = {c:i for i, c in enumerate(np.unique(array))}
    # print(mapper)
    # print(np.vectorize(lambda x: mapper[x])(array))
    return np.vectorize(lambda x: mapper[x])(array)

# TODO 未完成
def __concat_sensor_value(data):
    """加速度データと角加速度データをくっつける

    Parameters
    ----------
    data : Dict
        加速度データと角加速度データ
    """
    
    print("concat")
    
    df_a: pd.DataFrame = data['accelerometer']
    df_g: pd.DataFrame = data['gyro']

    # MEMO
    pd.concat([df_a.iloc[:10, 2], df_g.iloc[:10, 2]], axis=1).set_axis(["1","2"], axis=1)
    pd.concat([df_a.iloc[:10, 2], df_g.iloc[:10, 2]], axis=1) / 1e6
    df_a.iloc[:10, 2] - df_a.iloc[:10, 2].shift(1)
    df_a.loc[:, 'Index'].value_counts().sort_index()
    df_a[df_a.User == 'a'].loc[:, 'Index'].value_counts().sort_index()
    df_a[(df_a.User == 'a') & (df_a.Index == 0)]

    # Indexを基準にくっつける．一応Arrival_Time, Creation_Timeの確認をすること
    # くっつけるタイミングをどこにするか考える．おそらくセグメントに分けた後あたりがよさそう
    # 人とデバイスと行動が一致するように気を付けること
    # Indexが多少前後している可能性あり　29, 31, *30*, 32, 33
    # この関数では難しいことはせずに，一番細かくなっているDetaFrameの状態でくっつけれればいい
    # df_a[df_a.User & df_a.Device] == Indexの切れ目？ 0から始まらない計測もある感じ．．．

    raise NotImplementedError()
        
