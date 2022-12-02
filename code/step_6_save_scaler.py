import warnings
from typing import *

import os
import hydra
import pandas as pd
import joblib
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from step_1_pre_train_ml import load_data


# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(X, y, cv, cfg):
    
    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]
    y = y.astype(int)
    assert X.dtypes.value_counts().size == 1

    cv_idx = 0
    for train_idx, valid_idx, test_idx in cv.split(X, y):
        cv_idx += 1

        X_tr = X.iloc[train_idx]
        scaler = StandardScaler().fit(X_tr)
        save_name = f'{cfg.station_code}_cv{cv_idx}'
        joblib.dump(scaler, f'output/all_port_eachCV_scaler/{save_name}.h5')
        # joblib.dump(scaler, f'output/all_port_eachCV_scaler/{save_name}.pkl')

        X_tr_noTime = X.iloc[:,:-4]
        scaler2 = StandardScaler().fit(X_tr_noTime)
        joblib.dump(scaler2, f'output/all_port_eachCV_scaler/{save_name}_noTime.h5')


def _main(cfg: Union[DictConfig, OmegaConf]):

        for port in ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
                     'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']:
                
            cfg.station_code = port
            
            X, y, cv = load_data(cfg)
            custom_cross_val_predict(X, y, cv, cfg)


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

