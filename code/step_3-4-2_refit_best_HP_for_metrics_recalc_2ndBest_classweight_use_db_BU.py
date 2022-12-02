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

from pathlib import Path
from hydra.utils import get_original_cwd

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
def custom_cross_val_predict(X, y, cv, sample_weight, cfg, my_options, 
                             label1_weight) -> List:
    
    score_list = []
    fit_params = {}

    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]
    y = y.astype(int)
    assert X.dtypes.value_counts().size == 1

    # iterate over split
    cv_idx = 0
    model_allCV = []
    obs_pred_allCV = []
    for train_idx, valid_idx, test_idx in cv.split(X, y):
        cv_idx += 1
        model = load_pipe(cfg, label1_weight)  # loop안에서 정의해야 model cv독립적으로 생성됨
        
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
        
        fit_params[f"{model.steps[-1][0]}__sample_weight"] = sample_weight[train_idx]
        fit_params[f"{model.steps[-1][0]}__eval_set"] = [(X_val, y_val)]
        try:  # LGB에 is_unbalance같은 option주고 학습하면 가끔 error뜸.
            # 버전 업데이트 해도 생기고
            # 어쩔때는 안생기고, try except로 두번 시도하면 해결되는거 같기도 하고.
            # loss가 1도 안떨어지고 안올라갈때 발생하는 듯 함.
            model.fit(X_tr, y_tr, **fit_params)
        except:
            print(f'{cfg.model_name} crashed !!!!')
            print(f'{cfg.model_name} try one more time !!!!')
            model.fit(X_tr, y_tr, **fit_params)
            
        pred = model.predict(X_te)
        assert pred.ndim == 1

        # get test score
        selection_mask = np.logical_not(np.isnan(y_te))
        score = calc_metrics(y_te[selection_mask], pred[selection_mask])  # test set 결과 뽑는 것임
        score_list.append(score)
        df_debug.loc[f"{cfg.station_code}_{cfg.pred_hour}H_cv{cv_idx}"] = score

        obs_pred = y_te.copy().to_frame(name='obs')
        obs_pred.loc[y.index[test_idx], 'pred'] = pred
        # joblib save 전후 같은지 비교 : 같은듯
        # joblib.dump(model, './test.h5') 
        # model2 = joblib.load('./test.h5')
        # pred2 = model2.predict(X_te)
        # score2 = calc_metrics(y_te[selection_mask], pred2[selection_mask])
        # score
        # score['TP'] + score['FP'] + score['FN'] + score['TN']
        # score2['TP'] + score2['FP'] + score2['FN'] + score2['TN']
        model_allCV.append(model)
        obs_pred_allCV.append(obs_pred)
    
    # concatenate cross validation result and 
    # calc final result in macro fashion(1d numpy array) 
    metrics = ["ACC", "PAG", "POD", "F1"]
    arr = np.array(
            [score_list[i][metric] for metric in metrics 
                                   for i in range(len(score_list))])
    macro_score = np.mean(arr.reshape(-1,len(score_list)), axis=1)*100
    macro_score = macro_score.round(2)
    return macro_score.tolist(), model_allCV, obs_pred_allCV


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

    elif model_name == "xgb":
        return xgb.XGBClassifier(
            # scale_pos_weight = weight,
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
    my_options['save_best_model'] = True  #FIXME:
    my_options['save_model_pth'] = \
                        f'/home/sdh/fog-generation-ml/output/best_models_{exp}'
    my_options['save_each_time_cv'] = True  #FIXME:
    my_options['save_each_time_cv_pth'] = \
                    f'/home/sdh/fog-generation-ml/output/predResult_each_cv_{exp}'
    my_options['exp'] = exp

    df = pd.DataFrame(columns=["ACC", "PAG", "POD", "F1", 'model'], 
                      index=['1', '3', '6'])
    df_summary = pd.DataFrame(columns=['acc_1', 'acc_3', 'acc_6',
                                    'f1_1', 'f1_3', 'f1_6']).T
    with pd.ExcelWriter(
                        f'{save_dir}/all_port_bestScore_ml_'\
                        f'{my_options["exp"]}.xlsx') as excel_writer:

        for port in ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
                        'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']:
            for pred_hour in [1, 3, 6]:
                
                cfg.pred_hour = pred_hour
                cfg.station_code = port

                X, y, cv = load_data(cfg)

                best_f1 = []
                best_model_name = []
                for model in ['cb', 'lgb', 'rf', 'xgb']: 
                    target = '/'.join(cfg.log_prefix.split('/')[:4]) \
                                + f'/{port}/{pred_hour}/{model}'
                    one_config_file = (Path(get_original_cwd()) / target).glob(
                                                '**/.hydra/config.yaml').__next__()
                    metric_file = one_config_file.parent.parent.parent / \
                                                        'optimization_results.yaml'
                    f1 = OmegaConf.load(metric_file)['best_value']
                    best_f1.append(f1)
                    best_model_name.append(model)
                idx1 = best_f1.index(sorted(best_f1)[-1])
                idx2 = best_f1.index(sorted(best_f1)[-2])
                best_secBest = [best_model_name[idx1], best_model_name[idx2]]


                f1_best_models = []
                best_metrics_list = []
                best_2models_eachCV = []
                best2_obs_preds_eachCV = []
                for best_model_name in best_secBest:

                    # load best model
                    loaded_study = optuna.load_study(
                                study_name=f"{port}_{pred_hour}_{best_model_name}",
                                storage=f"sqlite:///{db_name}")
                    best_record = loaded_study.best_trial
                    best_f1 = best_record._values[0]
                    best_modelParams = best_record._params
                    best_modelParams = \
                            {item.split('.')[-1]:best_modelParams[item] 
                                                for item in best_modelParams}
                    # load HP of best model
                    cfg_bestmodel = OmegaConf.load(
                                            f'conf/model_cfg/{best_model_name}.yaml')    
                    cfg_bestmodel.pop('hydra')
                    cfg_bestmodel.pop('model_name')
                    
                    # get best params
                    for item in best_modelParams:
                        if item in cfg_bestmodel:
                            cfg_bestmodel[item] = best_modelParams[item]
                        elif item in cfg_bestmodel.model_params:
                            cfg_bestmodel.model_params[item] = \
                                                    best_modelParams[item]
                
                    mp = cfg_bestmodel.pop('model_params')
                    mp = {**cfg_bestmodel, **mp}
                    
                    # best param으로 업데이트하도록 위치 옮김
                    sample_weight = np.ones_like(y)
                    mask = (y == 1)
                    label1_weight = \
                        (len(y)-sum(y))/(sum(y) + 1) * cfg_bestmodel.pos_label_ratio
                    # give less weight in y==0 samples
                    sample_weight[~mask] = 1 / label1_weight  
                    
                    if best_model_name == 'cb':
                        mp.pop('pos_label_ratio')
                    
                    # finally update cfg (from hydra) variable
                    cfg.model_params = mp
                    cfg.model_name = best_model_name


                    # train and get 3fold test result
                    print(f'\n\n{"="*40}')
                    print(f'Now refit with best HP: {port}_{pred_hour}_{best_model_name}')
                    print(f'{"="*40}')

                    best_metrics, model, obs_pred = custom_cross_val_predict(
                        X, y, cv, sample_weight, cfg, my_options, label1_weight)

                    best_metrics_list.append(best_metrics)
                    f1_best_models.append(best_metrics[-1])
                    best_2models_eachCV.append(model)
                    best2_obs_preds_eachCV.append(obs_pred)

                # select best index
                idx = f1_best_models.index(max(f1_best_models))
                best_metrics = best_metrics_list[idx]
                best_model_name = best_secBest[idx]
                best_model = best_2models_eachCV[idx]
                obs_pred_best = best2_obs_preds_eachCV[idx]
                
                # Only best model !! save result model and prediction result
                if my_options['save_best_model']:
                    for cv_idx in range(0,3):
                        joblib.dump(best_model[cv_idx], 
                        f'{my_options["save_model_pth"]}/{cfg.station_code}_'\
                        f'{cfg.pred_hour}_best_{cfg.model_name}_cv{cv_idx+1}.h5') 

                if my_options['save_each_time_cv']:
                    for cv_idx in range(0,3):
                        obs_pred_best[cv_idx].to_csv(
                            f"{my_options['save_each_time_cv_pth']}/{cfg.station_code}_"\
                            f"{cfg.pred_hour}_best_{cfg.model_name}_cv{cv_idx+1}.csv"
                        )

                # macro fashion all scores
                df.loc[str(pred_hour),:] = list(best_metrics)+[best_model_name]
                df.to_excel(excel_writer, sheet_name=port, index_label=port, 
                            header=True)
                df_summary.loc[f'acc_{pred_hour}',port] = best_metrics[0]
                df_summary.loc[f'f1_{pred_hour}',port] = best_metrics[-1]
            # show one port-result
            print(df)

        df_summary.to_csv(
            f'{save_dir}/all_port_bestScore_ml_summary_{my_options["exp"]}.csv')
        df_debug.to_csv(
            f'{save_dir}/debug_{my_options["exp"]}.csv')
        

@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

