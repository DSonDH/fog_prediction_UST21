import warnings
from typing import *

import os
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
from utils import calc_metrics

warnings.simplefilter(action='ignore', category=FutureWarning)

#??? get best and second best model? 
# best랑 second best의 성능차이가 은근 심한 경우가 보임.
# 같은 boosting계열이라도 성능차이가 나니깐 여러 모델 실험하는 이유가 됨

# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(X, y, cv, sample_weight, cfg, my_options):
    
    score_list = []
    fit_params = {}

    # iterate over split
    cv_idx = 0
    for train_idx, valid_idx, test_idx in cv.split(X, y):
        cv_idx += 1
        model = load_pipe(cfg)  # model cv독립적으로 생성 

        usecols = [x for x in X.columns if x not in cfg.drop_cols]
        X = X[usecols]
        y = y.astype(int)
        assert X.dtypes.value_counts().size == 1
        
        obs = y.iloc[test_idx]
        if sum(obs) == 0:
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
            
        assert pred.ndim == 1, 'predition dimension is odd'
        selection_mask = np.logical_not(np.isnan(obs))
        score = calc_metrics(obs[selection_mask], pred[selection_mask])
        score_list.append(score)

        if not os.path.exists(my_options["save_model_pth"]):
            os.makedirs(my_options["save_model_pth"])
            os.makedirs(my_options['save_each_time_cv_pth'])

        # save model and prediction records
        joblib.dump(model, 
                f'{my_options["save_model_pth"]}/{cfg.station_code}_'\
                f'{cfg.pred_hour}_{cfg.model_name}_cv{cv_idx}.pkl') 
    
        obs_pred = obs.copy().to_frame(name='obs')
        obs_pred.loc[y.index[test_idx], 'pred'] = pred
        obs_pred.to_csv(
                f"{my_options['save_each_time_cv_pth']}/{cfg.station_code}_"\
                f"{cfg.pred_hour}_{cfg.model_name}_cv{cv_idx}.csv")


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
    choose_best = 2    
    my_options = {}
    my_options['save_best_model'] = True  #FIXME:
    my_options['save_model_pth'] = \
              f'/home/sdh/fog-generation-ml/output/for_Ensemble_best'\
              f'{choose_best}_ml_models'  #FIXME:
    my_options['save_each_time_cv'] = False  #FIXME:
    my_options['save_each_time_cv_pth'] = \
               f'{my_options["save_model_pth"]}/csvFiles'

    df_summary = pd.DataFrame(columns=['pred_1', 'pred_3', 'pred_6']).T
    for port in ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
                    'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']:
        for pred_hour in [1, 3, 6]:
            cfg.pred_hour = pred_hour
            cfg.station_code = port
            
            # data load
            X, y, cv = load_data(cfg)
            sample_weight = np.ones_like(y)
            mask = (y == 1)
            sample_weight[~mask] = 1 / cfg.pos_label_weights

            # select best models
            model_score = {}
            for model in ['cb', 'lgb', 'rf', 'xgb']: 
                loaded_study = optuna.load_study(
                        study_name=f"{port}_{pred_hour}_{model}",
                        storage=f"sqlite:///{db_name}")
                record = loaded_study.best_trial
                f1 = record._values[0]
                model_score[model] = f1
            df_ms = pd.DataFrame(model_score, index=['f1']).T
            df_ms = df_ms.sort_values(by=['f1'], ascending=False)
            
            # refit best models
            best_model_list = []
            for i in range(choose_best):
                best_model = df_ms.index[i]
                best_model_list.append(best_model)

                loaded_study = optuna.load_study(
                            study_name=f"{port}_{pred_hour}_{best_model}",
                            storage=f"sqlite:///{db_name}")
                best_record = loaded_study.best_trial
                best_f1 = best_record._values[0]
                best_modelParams = best_record._params
                best_modelParams = \
                        {item.split('.')[-1]:best_modelParams[item] 
                                            for item in best_modelParams}
                # load HP of best model
                cfg_bestmodel = OmegaConf.load(
                                        f'conf/model_cfg/{best_model}.yaml')    
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
                if best_model == 'cb':
                    mp = {**cfg_bestmodel, **mp}
                    mp.pop('pos_label_weights')
                
                # finaly update cfg (from hydra) variable
                cfg.model_params = mp
                cfg.model_name = best_model

                # train and get 3fold test result
                print(f'\n\n{"="*40}')
                print(f'Now refit with best HP: {port}_{pred_hour}_{best_model}')
                print(f'{"="*40}')
                
                # do refit and save model or csv file
                custom_cross_val_predict(
                                    X, y, cv, sample_weight, cfg, my_options)
                
            df_summary.loc[f'pred_{pred_hour}',port] = '_'.join(best_model_list)
            
    df_summary.to_csv(
            f'{my_options["save_model_pth"]}/Ensemble_ml_best{choose_best}.csv')


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

