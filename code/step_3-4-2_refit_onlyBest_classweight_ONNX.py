from typing import *

import os
import hydra
import numpy as np
from pathlib import Path
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from hydra.utils import get_original_cwd

import catboost
import lightgbm
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from step_1_pre_train_ml import load_data
from utils import calc_metrics

# onnx converting experiment
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

import onnxruntime as rt
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from skl2onnx.common.data_types import FloatTensorType, FloatType

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(X, y, cv, label1_weight, sample_weight, cfg)\
                             -> list:
    
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

        fit_params[f"{model.steps[-1][0]}__sample_weight"] = \
                                                        sample_weight[train_idx]
        fit_params[f"{model.steps[-1][0]}__eval_set"] = [(X_val, y_val)]

        # do fitting
        try:  # LGB에 is_unbalance같은 option주고 학습하면 가끔 error뜸.
            model.fit(X_tr, y_tr, **fit_params)
        except:
            print(f'{cfg.model_name} crashed !!!!')
            print(f'{cfg.model_name} try one more time !!!!')
            try:
                model.fit(X_tr, y_tr, **fit_params)
            except:
                print(f'{cfg.model_name} crashed !!!!')
                print(f'{cfg.model_name} try one more time !!!!')
                cfg.model_name = 'lgb_noOption'
                model = load_pipe(cfg, label1_weight)
                model.fit(X_tr, y_tr, **fit_params)

        # ONNX(Open Neural Network Exchange) converting and save
        # 발음은 onyx [오닉스] 라고 함.. ㅋㅋ
        if cfg.model_name != 'cb':

            # registration 과정이 필요하다나 
            if cfg.model_name in ['lgb', 'rf']:
                classifier = LGBMClassifier
                converter = convert_lightgbm
            elif cfg.model_name == 'xgb':
                classifier = XGBClassifier
                converter = convert_xgboost

            update_registered_converter(
                classifier, f'{cfg.model_name.upper()}Classifier',
                calculate_linear_classifier_output_shapes, converter,
                options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
            )
            # nocl : 아마도 number of classes 줄임말인듯. 
            # categorical 0 -> 0 ~ N_class-1까지 

            # The operator ZipMap produces a list of dictionaries. 
            # It repeats class names or ids but that’s not necessary (see issue 2149). 
            # By default, ZipMap operator is added, it can be deactivated by using: 
            # >>> options={type(model): {'zipmap': False}}
            
            """ FIXME: onnx로 converting시 차이가 생김.
                XOR이 없더라도, 인스턴스 별 예측 확률을 비교하면 다름!.
                tutorial에서 보여주는 수준의 정밀도가 아님

                try1: target opset 바꾸기 : 소용없음
                try2: converting 방법을 to_onnx로 : 소용없음
                try3: tutorial 데이터와 코드는 : 잘 됨
                      tutorial에 나온 모듈 버전과 내 버전 정확히 같음.
                      즉 데이터, 모델 tree 구조(?) 가 달라서 생기는 문제임
                try4: 내 모델로 iris 데이터 학습 : raw/onnx 차이 생김
                      즉 모델 구조가 달라져서 생기는 문제임
                try5: onnx version 관련 모든 옵션 수치 대입 소용없음
                      target_opset의 {'', 'ai.onnx.ml'} 두개 실험해봄
                      model_onnx에 실제 저장된 opset_import 보면 
                      domain: "ai.onnx.ml"
                      version: 3
                      domain: ""
                      version: 9
                      으로 고정되어있음. 아마도 그래프 생성하는 최소 도메인 버젼으로
                      고정하는 듯 함. 내가 아무리 높은 버전을 줘도.
                      즉, 모델 구조는 onnx에서 모두 구현을 하는데,
                      그 구현된게 틀린것이다 ?
            """
            model_onnx = convert_sklearn(
                model, 'pipeline',
                [('input', FloatTensorType([None, 186]))],
                # target_opset={'': 17}  # 17이 한계.
                target_opset={'': 17, 'ai.onnx.ml': 17}
            )
            
            # 바뀐거 저장
            save_name = f'./output/ONNX/{cfg.station_code}_{cfg.model_name}_'\
                        f'{cfg.pred_hour}H_cv{cv_idx}.onnx'
            with open(save_name, "wb") as f:
                f.write(model_onnx.SerializeToString())
            
            # onnx load          
            # 아래 에러는 문제 안되는 것으로 보임. 판별 결과가 달라지지 않거든.
            # UserWarning: X does not have valid feature names, but 
            # StandardScaler was fitted with feature names
            sess = rt.InferenceSession(save_name)

            if isinstance(X_te.astype(np.float32), pd.DataFrame):
                sess_input = X_te.astype(np.float32).to_numpy()
            else:
                sess_input = X_te.astype(np.float32)
            
            pred_onnx = sess.run(None, {'input': sess_input})

            # onnx 저장하고 불러오는데 차이는 없는 듯 함 (일부 샘플만 확인)
            # sess_raw = rt.InferenceSession(model_onnx.SerializeToString())
            # pred_raw_onnx = sess_raw.run(None, {'input': sess_input})

        else:  # catboost 전용 코드
            save_name = f'./output/ONNX/{cfg.station_code}_{cfg.model_name}_'\
                        f'{cfg.pred_hour}H_cv{cv_idx}.onnx'
            
            model[1].save_model(
                                save_name,
                                format="onnx",
                                export_parameters={
                                    'onnx_domain': 'ai.catboost',
                                    'onnx_model_version': 1,  #???
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

        # ONNX 모델 결과 다른지 확인
        pred_model = model.predict(sess_input)
        pred_model_prob = model.predict_proba(sess_input)
        print(
            f"{cfg.station_code}_{cfg.model_name}_{cfg.pred_hour}H_cv{cv_idx}" \
            f" : pred {sum(pred_model)} vs ONNX {sum(pred_onnx[0])}"
        )

        set1 = set([i for i in range(len(pred_model)) if pred_model[i] == 1])
        set2 = set([i for i in range(len(pred_onnx[0])) if pred_onnx[0][i] == 1])
        # set1_2 = set1 - set2
        XOR = set1 ^ set2
        if len(XOR):
            print( f'!!! mismatch BTW. origin model and ONNX model occured !!!')
            missing_idx = list(XOR)[0]
            print(f'{missing_idx} : {pred_model_prob[missing_idx]}')
            print(f'{missing_idx} : {pred_onnx[1][missing_idx]}')
            print()

        pred = model.predict(X_te)

    return 1


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
    exp = 'exp5'
    db_name = f'study_{exp}.db'
    save_dir = f'./output/ONNX'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


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


            for best_model in best_secBest:
                
                # load best model
                loaded_study = optuna.load_study(
                            study_name=f"{port}_{pred_hour}_{best_model}",
                            storage=f"sqlite:///{db_name}")
                best_record = loaded_study.best_trial
                # best_f1 = best_record._values[0]
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

                _ = custom_cross_val_predict(
                        X, y, cv, label1_weight, sample_weight, cfg)



@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

