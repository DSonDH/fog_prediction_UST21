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
    y = X.loc[:, 'vis_lag00'].copy()
    y.loc[:] = np.where( np.isnan(y),
                    np.nan,
                    np.where(np.less_equal(y, 1000),
                             2,
                             np.where(np.less(y, 3000), 1, 0)
                             )
                    ) 
            
    # test = pd.date_range('2018-01-01 00:00', '2022-06-30 23:50', freq='10T')
    # y_before = y.copy()
    y = y.shift(-1 * cfg.pred_hour * 6)

    # nan 제거 전 233047 : 3243 : 158
    # nan 제거 후 201174 : 2918 : 124

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


    
