from distutils.command.config import config
import warnings
from typing import *

import os
import catboost
import hydra
import lightgbm
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from step_1_pre_train_ml_3class import load_data
from utils import calc_metrics

from sklearn.metrics import multilabel_confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)

# metric of interest
MOI = ["ACC", "macro_PAG", "macro_POD", "macro_F1", 
                 "cm1", "cm2", "cm3", "cm4", "cm5", "cm6", "cm7", "cm8", "cm9"]
df1 = pd.DataFrame(columns=MOI)
df2 = pd.DataFrame(columns=["ACC", "macro_PAG", "macro_POD", "macro_F1"])

# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(X, y, cv, label0_weight, sample_weight, cfg):
    
    fit_params = {}

    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]
    y = y.astype(int)
    assert X.dtypes.value_counts().size == 1

    # iterate over split
    cv_idx = 0
    for train_idx, valid_idx, test_idx in cv.split(X, y):
        cv_idx += 1
        if cv_idx > 1:
            continue
        model = load_pipe(cfg, label0_weight)  # loop안에서 정의해야 model cv독립적으로 생성됨

        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[valid_idx]
        y_val = y.iloc[valid_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]

        fit_params[f"{model.steps[-1][0]}__sample_weight"] = sample_weight[train_idx]
        fit_params[f"{model.steps[-1][0]}__eval_set"] = [(X_val, y_val)]
        # do fitting
        try:  
            model.fit(X_tr, y_tr, **fit_params)
        except:
            print(f'{cfg.model_name} crashed !!!!')
            print(f'{cfg.model_name} try one more time !!!!')
            model.fit(X_tr, y_tr, **fit_params)
            
        pred = model.predict(X_te)

        # get test score
        selection_mask = np.logical_not(np.isnan(y_te))
        assert sum(~selection_mask) == 0
        score = calc_metrics(y_te, pred, binary=False)
        
        metrics = ["ACC", "macro_PAG", "macro_POD", "macro_F1"]
        for metric in metrics:
            score[metric] = np.round(score[metric] * 100, 2)
        
        df1.loc[f"{cfg.station_code}_{cfg.pred_hour}H_cv{cv_idx}"] = score
                
        mcm = multilabel_confusion_matrix(y_te, pred)
        for i in range(3):
            TN, FP, FN, TP = mcm[i].flatten()
            label = i
            acc = np.round((TN + TP) / (TN + FP + FN + TP) * 100, 2)
            pag = np.round((TP) / (TP + FP) * 100, 2)
            pod = np.round((TP) / (TP + FN) * 100, 2)
            f1 = np.round((2 * pag * pod) / (pag + pod), 2)
            df2.loc[f"{cfg.station_code}_{cfg.pred_hour}H_label{label}"] = \
                                                            [acc, pag, pod, f1]
        return score



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
            # scale_pos_weight = weight,
            **model_params,
        )

    elif model_name == "lgb":
        return lightgbm.LGBMClassifier(
            # is_unbalance = True,
            # scale_pos_weight = weight,
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
            # class_weights = [1, weight],
            **model_params
        )


# load scikit-learn pipeline
def load_pipe(cfg: DictConfig, weight):
    base_model = get_estimator(weight, cfg.model_name, cfg.model_params)
    pipe = make_pipeline(
        StandardScaler(),
        base_model,)
    return pipe


def _main(cfg: Union[DictConfig, OmegaConf]):
    exp = '3class'  #FIXME:
    db_name = f'study_{exp}.db'
    save_dir = f'./output/best_params_allModel_{exp}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    port = 'SF_0003'
    for pred_hour in [1, 3, 6]:
        
        cfg.pred_hour = pred_hour
        cfg.station_code = port
        
        X, y, cv = load_data(cfg)

        best_model = 'cb'
        
        # load best model
        loaded_study = optuna.load_study(
                       study_name=f"{port}_{pred_hour}_{best_model}",
                       storage=f"sqlite:///{db_name}"
                       )
        best_record = loaded_study.best_trial
        best_f1 = best_record._values[0]
        best_modelParams = best_record._params
        best_modelParams = {item.split('.')[-1]:best_modelParams[item] 
                            for item in best_modelParams}
        # load HP of best model
        cfg_bestmodel = OmegaConf.load(
                                f'conf/model_cfg_{exp}/{best_model}.yaml')    
        cfg_bestmodel.pop('hydra')
        cfg_bestmodel.pop('model_name')
        
        # update base config with best-hyperparameters from study.db
        for item in best_modelParams:
            if item in cfg_bestmodel:
                cfg_bestmodel[item] = best_modelParams[item]
            elif item in cfg_bestmodel.model_params:
                cfg_bestmodel.model_params[item] = best_modelParams[item]

        mp = cfg_bestmodel.pop('model_params')
        mp = {**cfg_bestmodel, **mp}

        # best param으로 업데이트하도록 위치 옮김
        sample_weight = np.ones_like(y)
        mask0 = (y == 0)
        mask1 = (y == 1)
        mask2 = (y == 2)
        
        label0_weight = (sum(mask0))/(sum(mask2) + 1) * cfg.pos_label_ratio
        sample_weight[mask0] = 1 / label0_weight  # give less weight

        if best_model == 'cb':
            mp.pop('pos_label_ratio')
        
        # finally update cfg (from hydra) variable
        cfg.model_params = mp
        cfg.model_name = best_model

        # train and get test result
        print(f'\n\n{"="*40}')
        print(f'Now refit with best HP: {port}_{pred_hour}_{best_model}')
        print(f'{"="*40}')

        best_metrics = custom_cross_val_predict(
                X, y, cv, label0_weight, sample_weight, cfg
        )
    
    df1.to_csv(f'{save_dir}/CM_macro_score_cb.csv')
    df2.to_csv(f'{save_dir}/each_label_score_cb.csv')

    

@hydra.main(config_path="../conf", config_name="config_3class.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

