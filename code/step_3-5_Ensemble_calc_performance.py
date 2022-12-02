from distutils.command.config import config
import warnings
from typing import *

import catboost
import hydra
import lightgbm
import numpy as np
import joblib
import optuna
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from step_1_pre_train_ml import load_data

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             jaccard_score, precision_score, recall_score)

warnings.simplefilter(action='ignore', category=FutureWarning)

"""
best랑 second best의 성능차이가 은근 심한 경우가 보임.
같은 boosting계열이라도 성능차이가 나니깐 여러 모델 실험하는 이유가 됨
"""

def custom_cross_val_predict(X, y, cv, sample_weight, cfg, my_options) -> float:
    obs_3cv = []
    pred_3cv = []
    fit_params = {}

    cv_idx = 0
    passed_cv_idx = []
    for train_idx, valid_idx, test_idx in cv.split(X, y):
        cv_idx += 1
        model = load_pipe(cfg)

        usecols = [x for x in X.columns if x not in cfg.drop_cols]
        X = X[usecols]
        y = y.astype(int)
        assert X.dtypes.value_counts().size == 1
        
        obs = y.iloc[test_idx]
        if sum(obs) == 0:
            passed_cv_idx.append(cv_idx)
            continue
        
        if cfg.model_name in ['cb', 'rf', 'lgb']:
            fit_params[f"{model.steps[-1][0]}__sample_weight"] = \
                    sample_weight[train_idx]
            # xgb 는 scale_pos_weight가 defalut로 적용됨

        # validation set 지정후 fit and test
        if cfg.model_name == 'xgb':
            X2 = X.values
            y2 = y.values
            fit_params[f"{model.steps[-1][0]}__eval_set"] = \
                    [(X2[valid_idx], y2[valid_idx])]
            model.fit(X2[train_idx], y2[train_idx], **fit_params)
            pred = model.predict(X2[test_idx])
            
        else:
            fit_params[f"{model.steps[-1][0]}__eval_set"] = \
                    (X.iloc[valid_idx], y.iloc[valid_idx])
            model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_params)
            pred = model.predict(X.iloc[test_idx])
        
        obs_3cv.append(obs)
        pred_3cv.append(pred)
        
    return obs_3cv, pred_3cv, passed_cv_idx


# get estimator
def get_estimator(model_name, model_params: Optional[Dict] = None):
    if model_params is None:
        model_params = {}

    if model_name == "rf":    
        return lightgbm.LGBMClassifier(
            **model_params,)

    elif model_name == "lgb":
        return lightgbm.LGBMClassifier(
            **model_params, )

    elif model_name == "cb":
        return catboost.CatBoostClassifier(
            **model_params, )

    elif model_name == "xgb":
        return xgb.XGBClassifier(
            **model_params,)


# load scikit-learn pipeline
def load_pipe(cfg: DictConfig):
    base_model = get_estimator(cfg.model_name, cfg.model_params)
    pipe = make_pipeline(
        StandardScaler(),
        base_model,)
    return pipe


def _main(cfg: Union[DictConfig, OmegaConf]):
    db_name = 'study_exp3.db'  #FIXME:
    save_dir = './output/best_params_allModel'
    models = ['cb', 'lgb', 'rf', 'xgb']

    my_options = {}
    my_options['save_best_model'] = False  #FIXME:
    my_options['save_model_pth'] = \
                        '/home/sdh/fog-generation-ml/output/best_models'
    my_options['save_each_time_cv'] = False  #FIXME:
    my_options['save_each_time_cv_pth'] = \
                        '/home/sdh/fog-generation-ml/output/predResult_each_cv'
    my_options['exp'] = 'exp1'  #FIXME:

    df = pd.DataFrame(columns=["ACC", "PAG", "POD", "F1", 'model'], 
                      index=['1', '3', '6'])
    df_summary = pd.DataFrame(columns=['acc_1', 'acc_3', 'acc_6',
                                    'f1_1', 'f1_3', 'f1_6']).T
    with pd.ExcelWriter(
                        f'{save_dir}/all_port_Ensemble_'\
                        f'{my_options["exp"]}.xlsx') as excel_writer:

        for port in ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
                     'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']:
            for pred_hour in [1, 3, 6]:
                
                cfg.pred_hour = pred_hour
                cfg.station_code = port
                
                X, y, cv = load_data(cfg)
                sample_weight = np.ones_like(y)
                mask = (y == 1)
                sample_weight[~mask] = 1 / cfg.pos_label_weights
                
                votes_cv1 = []
                votes_cv2 = []
                votes_cv3 = []
                for model in models: 
                    # load model
                    loaded_study = optuna.load_study(
                                study_name=f"{port}_{pred_hour}_{model}",
                                storage=f"sqlite:///{db_name}")
                    best_record = loaded_study.best_trial
                    best_modelParams = best_record._params
                    best_modelParams = \
                            {item.split('.')[-1]:best_modelParams[item] 
                                                for item in best_modelParams}
                    # load HP of best model
                    cfg_bestmodel = OmegaConf.load(
                                            f'conf/model_cfg/{model}.yaml')    
                    cfg_bestmodel.pop('hydra')
                    cfg_bestmodel.pop('model_name')
                    
                    # update base config with best-hyperparameters from study.db
                    for item in best_modelParams:
                        if item in cfg_bestmodel:
                            cfg_bestmodel[item] = best_modelParams[item]
                        elif item in cfg_bestmodel.model_params:
                            cfg_bestmodel.model_params[item] = \
                                                    best_modelParams[item]
                
                    mp = cfg_bestmodel.pop('model_params')
                    if model == 'cb':
                        mp = {**cfg_bestmodel, **mp}
                        mp.pop('pos_label_weights')
                    
                    # finaly update cfg (from hydra) variable
                    cfg.model_params = mp
                    cfg.model_name = model

                    # train and get 3fold test result
                    print(f'\n\n{"="*40}')
                    print(f'Now refit with best HP: {port}_{pred_hour}_{model}')
                    print(f'{"="*40}')
                    obs_3cv, pred_3cv, passed_cv = custom_cross_val_predict(
                                        X, y, cv, sample_weight, cfg, my_options
                    )

                    # cv fold마다 test길이가 다르므로 따로
                    # 결측있으면 합치지 말라고
                    if not 1 in passed_cv:
                        votes_cv1.append(pred_3cv[0])
                    if not 2 in passed_cv:
                        votes_cv2.append(pred_3cv[1])
                    if not 3 in passed_cv:
                        votes_cv3.append(pred_3cv[2])
                    obs = obs_3cv  # 모델마다 GT가 바뀌진 않으니까
                
                # append된 만큼 나누기 근데 한 cv가 결측이면 모든 모델에서 결측임.
                ensemble_cv1 = np.array(votes_cv1).sum(axis=0)/len(votes_cv1)
                ensemble_cv2 = np.array(votes_cv2).sum(axis=0)/len(votes_cv2)
                ensemble_cv3 = np.array(votes_cv3).sum(axis=0)/len(votes_cv3)

                if np.isnan(ensemble_cv3).any():
                    yhat = (ensemble_cv1 >= 0.5).astype(float),\
                        (ensemble_cv2 >= 0.5).astype(float)
                else:
                    yhat = (ensemble_cv1 >= 0.5).astype(float),\
                        (ensemble_cv2 >= 0.5).astype(float),\
                        (ensemble_cv3 >= 0.5).astype(float)
                
                best_metrics = []
                for i in range(len(yhat)):
                    y = obs[i]
                    yhat_cv = yhat[i]
                    if len(confusion_matrix(y, yhat_cv).flatten()) == 1 and sum(y) == 0:
                        print(f'!!! {port}_{pred_hour}_{cv}: all y and all pred was no-fog')
                        df_summary.loc[len(df_summary),:] = \
                                                        port, pred_hour, '', '', '', ''
                        continue

                    ACC = np.round(accuracy_score(y, yhat_cv),2)
                    PAG = np.round(precision_score(y, yhat_cv, average = 'macro'),2)
                    POD = np.round(recall_score(y, yhat_cv, average = 'macro'),2)
                    F1 = np.round(f1_score(y, yhat_cv, average = 'macro'),2)
                    best_metrics.append([ACC, PAG, POD, F1])

                # calc 3cv macro-averaged score
                best_metrics = np.array(best_metrics).mean(axis=0)
                models_str = '_'.join(models)

                # macro fashion all scores
                df.loc[str(pred_hour),:] = list(best_metrics)+[models_str]
                df.to_excel(excel_writer, sheet_name=port, index_label=port, 
                            header=True)
                df_summary.loc[f'acc_{pred_hour}',port] = best_metrics[0]
                df_summary.loc[f'f1_{pred_hour}',port] = best_metrics[-1]
            
            # show current result
            print(df)
        df_summary.to_csv(
            f'{save_dir}/all_port_Ensemble_summary_{my_options["exp"]}.csv')


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

