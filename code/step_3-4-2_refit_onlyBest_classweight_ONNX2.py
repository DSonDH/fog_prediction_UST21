import warnings
from typing import *

import os
import catboost
import hydra
import lightgbm
import numpy as np
from pathlib import Path
import optuna
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from step_1_pre_train_ml import load_data
from utils import calc_metrics

from hydra.utils import get_original_cwd

# onnx converting experiment
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

import onnxmltools
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
import onnxmltools.convert.common.data_types
# catboost 는 따로 해야되는듯 ?


import onnx
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

import onnxruntime as rt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail

import skl2onnx
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from skl2onnx.common.data_types import FloatTensorType, FloatType


warnings.simplefilter(action='ignore', category=FutureWarning)


# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(\
        X, y, cv, label1_weight, sample_weight, cfg) -> list:
    
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

        if sum(y_te) == 0 :
            print(f'!!! there is no fog label in this split')
            continue

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



                
        # TODO: ONNX(Open Neural Network Exchange) converting and save
        if cfg.model_name != 'cb':

            # registration 과정이 필요하다나 
            if cfg.model_name == 'lgb':
                classifier = LGBMClassifier
                converter = convert_lightgbm
            elif cfg.model_name == 'rf':
                classifier = LGBMClassifier
                converter = convert_lightgbm
            elif cfg.model_name == 'xgb':
                classifier = XGBClassifier
                converter = convert_xgboost

            update_registered_converter(
                classifier, f'{cfg.model_name.upper()}Classifier',
                calculate_linear_classifier_output_shapes, converter,
                options={'nocl': [True, False], 
                         'zipmap': [True, False, 'columns']
                        }
            )
            # nocl : 아마도 number of classes 줄임말인듯. categorical 0 -> 0 ~ N_class-1까지 

            # The operator ZipMap produces a list of dictionaries. 
            # It repeats class names or ids but that’s not necessary (see issue 2149). 
            # By default, ZipMap operator is added, it can be deactivated by using: 
            # >>> options={type(model): {'zipmap': False}}
            
            if isinstance(X_tr, pd.DataFrame):
                sess_tr_input = X_tr.astype(np.float32).to_numpy()
            else:
                sess_tr_input = X_tr.astype(np.float32)

            model_onnx = to_onnx(
                model, X=sess_tr_input,
                target_opset={'': 12, 'ai.onnx.ml': 2},
                # options={id(model): {'zipmap': False}}   
            )
            # model, 
            # X=None, 
            # name=None, 
            # initial_types=None,
            # target_opset=None, 
            # options=None,
            # white_op=None, 
            # black_op=None, 
            # final_types=None,
            # dtype=None, 
            # naming=None, 
            # verbose=0

            # 바뀐거 저장
            save_name = f'./output/ONNX/{cfg.station_code}_{cfg.model_name}_'\
                        f'{cfg.pred_hour}H_cv{cv_idx}.onnx'
            with open(save_name, "wb") as f:
                f.write(model_onnx.SerializeToString())
            
            # onnx load
            if isinstance(X_te.astype(np.float32), pd.DataFrame):
                sess_input = X_te.astype(np.float32).to_numpy()
            else:
                sess_input = X_te.astype(np.float32)
            
            # 아래 에러는 문제 안되는 것으로 보임. 판별 결과가 달라지지 않거든.
            # UserWarning: X does not have valid feature names, but 
            # StandardScaler was fitted with feature names

            sess = rt.InferenceSession(save_name)
            # sess = rt.InferenceSession(model_onnx.SerializeToString())
            pred_onnx = sess.run(None, {'X': sess_input})
            # ??? 1개씩 ?

        else:  # catboost 전용 코드
            save_name = f'./output/ONNX/{cfg.station_code}_{cfg.model_name}_'\
                        f'{cfg.pred_hour}H_cv{cv_idx}.onnx'
            
            model[1].save_model(
                                save_name,
                                format="onnx",
                                export_parameters={
                                    'onnx_domain': 'ai.catboost',
                                    'onnx_model_version': 12,  #???
                                    # 'onnx_doc_string': 'test model for MultiClassification',
                                    # 'onnx_graph_name': 'CatBoostModel_for_MultiClassification'
                                }
            )
            
            # catboost는 pipeline 저장이 안되므로 scaler 따로 실행해줘야 함
            scaler = StandardScaler()
            sess_input = scaler.fit_transform(X_te.to_numpy())

            sess = rt.InferenceSession(save_name)
            pred_onnx = sess.run(
                                ['label', 'probabilities'],
                                {'features': sess_input.astype(np.float32)}
                        )

        # 원본 prediction 결과와 ONNX 저장 모델 결과가 다름
        #!!! 입력값은 같은데, 원본 model prediction 결과만 달라지네? 
        pred_model = model.predict(sess_input)
        pred_model_prob = model.predict_proba(sess_input)
        print(
            f"{cfg.station_code}_{cfg.model_name}_{cfg.pred_hour}H_cv{cv_idx}"\
            f" : pred {sum(pred_model)} vs ONNX {sum(pred_onnx[0])}")
        
        set1 = set([i for i in range(len(pred_model)) if pred_model[i] == 1])
        set2 = set([i for i in range(len(pred_onnx[0])) if pred_onnx[0][i] == 1])
        # set1_2 = set1 - set2
        # set2_1 = set2 - set1
        
        XOR = set1 ^ set2
        if len(XOR):
            print( f'!!! mismatch BTW. origin model and ONNX model occured !!!')
            missing_idx = list(XOR)[0]
            print(f'{missing_idx} : {pred_model_prob[missing_idx]}')
            print(f'{missing_idx} : {pred_onnx[1][missing_idx]}')
            print()

        pred = model.predict(X_te)
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

    df = pd.DataFrame(columns=["ACC", "PAG", "POD", "F1", 'model'], 
                      index=['1', '3', '6'])

    for port in ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
                    'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']:
        for pred_hour in [1, 3, 6]:
            
            cfg.pred_hour = pred_hour
            cfg.station_code = port

            X, y, cv = load_data(cfg)

            best_f1 = []
            best_model = []
            for model in ['cb', 'lgb', 'rf', 'xgb']: 
                target = '/'.join(cfg.log_prefix.split('/')[:4]) \
                             + f'/{port}/{pred_hour}/{model}'
                one_config_file = (Path(get_original_cwd()) / target).glob(
                                            '**/.hydra/config.yaml').__next__()
                metric_file = one_config_file.parent.parent.parent / \
                                                     'optimization_results.yaml'
                f1 = OmegaConf.load(metric_file)['best_value']
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
                label1_weight = \
                    (len(y)-sum(y))/(sum(y) + 1) * cfg_bestmodel.pos_label_ratio
                
                sample_weight[~mask] = 1 / label1_weight  
                # give less weight in y==0 samples
                
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
                        X, y, cv, label1_weight, sample_weight, cfg)

                best_metrics_list.append(best_metrics)
                f1_best_models.append(best_metrics[-1])
            
            # finally select best model
            idx = f1_best_models.index(max(f1_best_models))
            best_metrics = best_metrics_list[idx]
            best_model = best_secBest[idx]
            
            # macro fashion all scores
            df.loc[str(pred_hour),:] = list(best_metrics)+[best_model]
        # show one port-result
        print(df)


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

