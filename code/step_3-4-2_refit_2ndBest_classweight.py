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

n_ftIPT = 30
df_ft_impt = pd.DataFrame(index = np.arange(n_ftIPT))


# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(\
        X, y, cv, label1_weight, sample_weight, cfg, my_options) -> list:
    
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

        # if sum(y_te) == 0 :
        #     print(f'!!! there is no fog label in this split')
        #     continue

        # TODO: 실제로 weight 적용되는건지, 모델 호출할 때 명시하는것과 다른건지 ?
        fit_params[f"{model.steps[-1][0]}__sample_weight"] = \
                                                        sample_weight[train_idx]
        # do fitting
        fit_params[f"{model.steps[-1][0]}__eval_set"] = [(X_val, y_val)]
        try:  # LGB에 is_unbalance같은 option주고 학습하면 가끔 error뜸.
            # 버전 업데이트 해도 생기고
            # 어쩔때는 안생기고, try except로 두번 시도하면 해결되는거 같기도 하고.
            # loss가 1도 안떨어지고 안올라갈때 발생하는 듯 함.
            model.fit(X_tr, y_tr, **fit_params)
        except:
            print(f'{cfg.model_name} crashed !!!!')
            print(f'{cfg.model_name} try one more time !!!!')
            try:  # LGB에 is_unbalance같은 option주고 학습하면 가끔 error뜸.
                model.fit(X_tr, y_tr, **fit_params)
            except:
                print(f'{cfg.model_name} crashed !!!!')
                print(f'{cfg.model_name} try one more time !!!!')
                cfg.model_name = 'lgb_noOption'
                model = load_pipe(cfg, label1_weight)
                model.fit(X_tr, y_tr, **fit_params)

        pred = model.predict(X_te)
        if ((cfg.model_name != 'SF_0010') and (cv_idx == 3)) or \
                ((cfg.model_name == 'SF_0010') and (cv_idx == 2)):
            note = f'{cfg.station_code}_{cfg.pred_hour}_{cfg.model_name}'

            if cfg.model_name == 'cb':
                ft_importance = model.steps[-1][-1].get_feature_importance()
                impt_mask = ft_importance > 0
                items = X.columns[impt_mask]
                values = np.round(ft_importance[impt_mask]/100, 4)
            elif cfg.model_name in ['lgb', 'rf']:
                ft_importance = model.steps[-1][-1].feature_importances_
                impt_mask = ft_importance > 0
                items = X.columns[impt_mask]
                values = np.round(ft_importance[impt_mask]/sum(impt_mask), 4)
            
            elif cfg.model_name == 'xgb':
                ft_importance = model.steps[-1][-1].feature_importances_
                impt_mask = ft_importance > 0
                items = X.columns[impt_mask]
                values = np.round(ft_importance[impt_mask], 4)

            if len(items) > n_ftIPT:
                ft_dict = {items[i]:values[i] for i in range(len(values))}
                ft_dict = {k: v for k, v in sorted(ft_dict.items(), key=lambda item: item[1], reverse=True)}
                ft_dict2 = {list(ft_dict.keys())[i]: list(ft_dict.values())[i] for i in range(n_ftIPT)}
                items = list(ft_dict2.keys())
                values = list(ft_dict2.values())
            df_ft_impt.loc[:len(values)-1, note + '_items'] = items
            df_ft_impt.loc[:len(values)-1, note + '_values'] = values

                                                      
        if my_options['save_best_model']:
            joblib.dump(model, 
              f'{my_options["save_model_pth"]}/{cfg.station_code}_'\
              f'{cfg.pred_hour}_best_{cfg.model_name}_cv{cv_idx}.h5') 
        if my_options['save_each_time_cv']:
            obs_pred = y_te.copy().to_frame(name='obs')
            obs_pred.loc[y.index[test_idx], 'pred'] = pred
            obs_pred.to_csv(
                f"{my_options['save_each_time_cv_pth']}/{cfg.station_code}_"\
                f"{cfg.pred_hour}_best_{cfg.model_name}_cv{cv_idx}.csv"
            )
        
        # get test score
    #     selection_mask = np.logical_not(np.isnan(y_te))
    #     score = calc_metrics(y_te[selection_mask], pred[selection_mask])  # test set 결과 뽑는 것임
    #     score_list.append(score)
    #     df_debug.loc[f"{cfg.station_code}_{cfg.pred_hour}H_cv{cv_idx}"] = score
    
    # # concatenate cross validation result and 
    # # calc final result in macro fashion(1d numpy array) 
    # metrics = ["ACC", "PAG", "POD", "F1"]
    # arr = np.array(
    #         [score_list[i][metric] for metric in metrics 
    #                                for i in range(len(score_list))])
    # macro_score = np.mean(arr.reshape(-1,len(score_list)), axis=1)*100
    # macro_score = macro_score.round(2)
    # return macro_score.tolist()
    return 0


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

        
            # ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
            #          'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']
        for port in ['SF_0011']:
            for pred_hour in [1, 3, 6]:
                
                cfg.pred_hour = pred_hour
                cfg.station_code = port
                
                X, y, cv = load_data(cfg)

                best_f1 = []
                best_model = []

                #FIXME: db 파일이 손상된 경우 data/log로 불러와야함
                !!!!

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
                    mp = {**cfg_bestmodel, **mp}

                    # best param으로 업데이트하도록 위치 옮김
                    sample_weight = np.ones_like(y)
                    mask = (y == 1)
                    label1_weight = (len(y)-sum(y))/(sum(y) + 1) * cfg_bestmodel.pos_label_ratio
                    sample_weight[~mask] = 1 / label1_weight  # give less weight in y==0 samples
                    
                    if best_model == 'cb':
                        mp.pop('pos_label_ratio')
                    
                    # finally update cfg (from hydra) variable
                    cfg.model_params = mp
                    cfg.model_name = best_model

                    # train and get 3fold test result
                    print(f'\n\n{"="*40}')
                    print(f'Now refit with best HP: {port}_{pred_hour}_{best_model}')
                    print(f'{"="*40}')

                    best_metrics = custom_cross_val_predict(
                            X, y, cv, label1_weight, sample_weight, cfg, my_options)

        #             best_metrics_list.append(best_metrics)
        #             f1_best_models.append(best_metrics[-1])
                
        #         idx = f1_best_models.index(max(f1_best_models))
        #         best_metrics = best_metrics_list[idx]
        #         best_model = best_secBest[idx]
                
        #         # macro fashion all scores
        #         df.loc[str(pred_hour),:] = list(best_metrics)+[best_model]
        #         df.to_excel(excel_writer, sheet_name=port, index_label=port, 
        #                     header=True)
        #         df_summary.loc[f'acc_{pred_hour}',port] = best_metrics[0]
        #         df_summary.loc[f'f1_{pred_hour}',port] = best_metrics[-1]
        #     # show one port-result
        #     print(df)

        # df_summary.to_csv(
        #     f'{save_dir}/all_port_bestScore_ml_summary_{my_options["exp"]}.csv')
        # df_debug.to_csv(
        #     f'{save_dir}/debug_{my_options["exp"]}.csv')
        # df_ft_impt.T.to_csv(
        #     f'{save_dir}/feature_importance_{my_options["exp"]}.csv')


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

