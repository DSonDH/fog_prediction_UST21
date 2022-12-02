from pathlib import Path
from typing import *

import joblib
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
# from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
#                            binary_opening)
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold


class Custom_cv_dh:
    def __init__(self, n_splits: Optional[int]=5):
        self.n_splits = 3
        self.cv_periods = [
            ['2021-07-01', '2022-06-30'],  
            ['2020-07-01', '2021-06-30'],
            ['2019-07-01', '2020-06-30'],
        ]  # 한 연도의 해무 빈도에 따라 성능 편차가 크므로 3년치 test해서 평균냄
        self.gss = StratifiedGroupKFold(n_splits=n_splits)

    def split(self, X, y, **kwargs):      
        for period in self.cv_periods:
            # TODO: tr val test분리 시 lagging이 서로 걸치면 안될거 같은데
            #  major problem은 아니겠지
            test_mask = np.logical_and(
                period[0] < y.index,
                y.index <= period[1])
            X_train = X[~test_mask]
            y_train = y[~test_mask]
            
            test_idx = np.flatnonzero(test_mask)
            trVal_idx = np.flatnonzero(~test_mask)

            groups = y_train.index.year * 366 + y_train.index.dayofyear  # 윤달 곂치게 안할려고
            train_idx, valid_idx = list(self.gss.split(X_train, y_train, 
                                                        groups=groups))[0]

            train_idx = trVal_idx[train_idx]
            valid_idx = trVal_idx[valid_idx]

            # assert (set(train_idx) & set(valid_idx)) == 0
            # assert (set(train_idx) & set(test_idx)) == 0

            yield train_idx, valid_idx, test_idx


def load_data(cfg: Union[DictConfig, OmegaConf]):

    station_code = cfg.station_code
    source = Path(get_original_cwd()) / cfg.catalogue.processed
    source = source.with_name(f'{station_code}.pkl')

    X = joblib.load(source)["x"]    
    X = X.set_axis([f"{c1}_{c2}" for c1, c2 in X.columns], axis=1)
    y = joblib.load(source)["y"].reset_index()
    y = y.set_index("datetime")
    X.index = y.index

    X = X.join(
        y[["time_day_sin",
                "time_day_cos",
                "time_year_sin",
                "time_year_cos",
                # "std_ws",
                # "std_ASTD",
                # "std_rh"
        ]]
    )
    y = y[cfg.target_name]
    drop_mask = X.isna().any(axis=1) | y.isna()
    X = X.iloc[np.flatnonzero(~drop_mask)]  # 46747여개 떨어져나감
    y = y[~drop_mask]
    assert not X.isna().sum().sum()    
    
    if cfg.stage == 'train':
        cv = Custom_cv_dh()
        return X, y, cv    
    else:
        test_start = cfg.test_start
        test_mask = y.index >= test_start
        usecols = [x for x in X.columns if x not in cfg.drop_cols]
        if cfg.stage == 'test':
            return X.loc[test_mask, usecols], y[test_mask]
        elif cfg.stage == 'refit':
            return X.loc[~test_mask, usecols], y[~test_mask]


# 고려대식 라벨 스무딩 방법. 해무 사이의 비해무는 해무로 처리하고, 동떨어진 해무는 비해무로 처리
def transform_korea(vis: pd.Series) -> pd.Series:
    assert vis.index.to_series().diff().value_counts().n_unique() == 0
    # 해무이면서 최근 1시간 동안 해무 데이터가 2개 이상인 데이터 mask
    end_points = np.logical_and(vis.eq(1), vis.rolling(6).sum().ge(2))
    end_points = end_points[end_points].index.sort_values()

    # 해무이면서 향후 1시간 동안 해무 데이터가 2개 이상인 데이터 mask
    start_points = np.logical_and(
        vis.sort_index(ascending=False).eq(1),
        vis.sort_index(ascending=False).rolling(6).sum().ge(2),
    )

    start_points = start_points[start_points].index.sort_values()
    assert (end_points - start_points).max() <= pd.Timedelta(minutes=50)

    # 위 둘 사이를 해무로 채워 넣음
    for start_point, end_point in zip(start_points, end_points):
        assert start_point <= end_point
        vis[
            start_point:end_point
        ] = 1  # start_point, end_point 둘 다 어차피 1이라서 closed | open interval 상관없음

    drop_masks = np.logical_and(
        vis.sort_index(ascending=False).eq(1),
        vis.sort_index(ascending=False)
        .rolling(11, center=True)
        .sum()
        .eq(1),  # 0 0 0 0 0 1 0 0 0 0 0 경우 drop mask
    )

    drop_masks = drop_masks[drop_masks].index
    vis[drop_masks] = 0
    return vis

if __name__ == '__main__':
    essential_folderes = [
        'data/model',
        'data/processed',
        'data/hparams',
        'data/clean',
        'data/model_out',
        'data/log'
    ]
    for folder in essential_folderes:
        Path(folder).mkdir(parents=True, exist_ok=True)

    
