from distutils.command.config import config
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
df_debug = pd.DataFrame(
        columns=["ACC", "PAG", "POD", "F1", "TN", "FP", "FN", "TP"])


# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(X, y, cv, sample_weight, cfg, weight2) -> float:
    
    score_list = []
    fit_params = {}

    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]
    y = y.astype(int)
    assert X.dtypes.value_counts().size == 1

    # iterate over split
    cv_idx = 0
    for train_idx, valid_idx, test_idx in cv.split(X, y):
        cv_idx += 1
        model = load_pipe(cfg, weight2)  # loop안에서 정의해야 model cv독립적으로 생성됨
        
        if cfg.model_name == 'xgb':
            X_tr = X.iloc[train_idx].values
            y_tr = y.iloc[train_idx].values
            X_val = X.iloc[valid_idx].values
            y_val = y.iloc[valid_idx].values
            X_te = X.iloc[test_idx].values
            y_te = y[test_idx]

        else:
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X.iloc[valid_idx]
            y_val = y.iloc[valid_idx]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]

        if sum(y_te) == 0 :
            print(f'!!! there is no fog label in this split')
            continue
        
        # do fitting
        fit_params[f"{model.steps[-1][0]}__eval_set"] = [(X_val, y_val)]

        model.fit(X_tr, y_tr, **fit_params)
            
        pred = model.predict(X_te)
        assert pred.ndim == 1

        # get test score
        selection_mask = np.logical_not(np.isnan(y_te))
        score = calc_metrics(y_te[selection_mask], pred[selection_mask])  # test set 결과 뽑는 것임
        score_list.append(score)
                                                        

    # concatenate cross validation result and 
    # calc final result in macro fashion(1d numpy array) 
    metrics = ["ACC", "PAG", "POD", "F1"]
    arr = np.array(
            [score_list[i][metric] for metric in metrics 
                                   for i in range(len(score_list))])
    macro_score = np.mean(arr.reshape(-1,len(score_list)), axis=1)*100
    macro_score = macro_score.round(2)
    return macro_score.tolist()


# get estimator
def get_estimator(weight, model_name, model_params: Optional[Dict] = None):
    if model_params is None:
        model_params = {}

    if model_name == "rf":    
        return lightgbm.LGBMClassifier(
            #  gbdt, traditional Gradient Boosting Decision Tree.
            #  dart, Dropouts meet Multiple Additive Regression Trees.
            #  goss, Gradient-based One-Side Sampling.
            #  rf, Random Forest.
            # is_unbalance = True,
            scale_pos_weight = weight,
            **model_params,
        )

    elif model_name == "lgb":
        return lightgbm.LGBMClassifier(
            # is_unbalance = True,
            scale_pos_weight = weight,
            **model_params
        )
    elif model_name == "lgb_noOption":
        return lightgbm.LGBMClassifier(
            # is_unbalance = True,
            # scale_pos_weight = weight,
            **model_params
        )


    elif model_name == "cb":
        return catboost.CatBoostClassifier(
            class_weights = [1, weight],
            **model_params
        )

    elif model_name == "xgb":
        return xgb.XGBClassifier(
            scale_pos_weight = weight,
            **model_params,
        )


# load scikit-learn pipeline
def load_pipe(cfg: DictConfig, weight):
    base_model = get_estimator(weight, cfg.model_name, cfg.model_params)
    pipe = make_pipeline(
        StandardScaler(),
        base_model,)
    return pipe


def _main(cfg: Union[DictConfig, OmegaConf]):
    exp = 'exp5'  #FIXME:
    db_name = f'study_{exp}.db'
    save_dir = f'./output/best_params_allModel_{exp}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    my_options = {}
    my_options['exp'] = exp

    
    # ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
    #              'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']
    for port in ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
                    'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']:
        for pred_hour in [1, 3, 6]:
            
            cfg.pred_hour = pred_hour
            cfg.station_code = port
            
            X, y, cv = load_data(cfg)
            sample_weight = np.ones_like(y)
            mask = (y == 1)
            # sample_weight[~mask] = 1 / cfg.pos_label_weights
            weight2 = len(y)-sum(y)/(sum(y) + 1)
            
            best_f1 = []
            best_model = []
            for model in ['cb', 'lgb', 'rf', 'xgb']: 
            # 기존 : 어차피 성능 덜 좋은 모델들은 best모델에 씹혀먹힘
            # 현재 : 항 별로 따로 성능 기록하여 best f1비교하여 best만 가져다 씀
                loaded_study = optuna.load_study(
                        study_name=f"{port}_{pred_hour}_{model}",
                        storage=f"sqlite:///{db_name}")
                record = loaded_study.best_trial
                f1 = record._values[0]
                best_f1.append(f1)
                best_model.append(model)
            idx1 = best_f1.index(sorted(best_f1)[-1])
            idx2 = best_f1.index(sorted(best_f1)[-2])
            best_secBest = [best_model[idx1], best_model[idx2]]
                
            f1_best_models = []
            best_metrics_list = []
            for best_model in best_secBest:
                        
                # load best model
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
                    mp.pop('pos_label_ratio')
                
                # finaly update cfg (from hydra) variable
                cfg.model_params = mp
                cfg.model_name = best_model

                # train and get 3fold test result
                print(f'\n\n{"="*40}')
                print(f'Now refit with best HP: {port}_{pred_hour}_{best_model}')
                print(f'{"="*40}')
                best_metrics = custom_cross_val_predict(
                        X, y, cv, sample_weight, cfg, my_options, weight2
                )



@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

