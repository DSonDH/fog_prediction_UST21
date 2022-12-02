import warnings
from typing import *

import catboost
import hydra
import lightgbm
import numpy as np
import xgboost as xgb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from step_1_pre_train_ml import load_data
from utils import calc_metrics

warnings.simplefilter(action='ignore', category=FutureWarning)


# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(X, y, cv, sample_weight, cfg) -> float:
    """
    model, x, y, cross validator, fit_params가 주어지면
    cv로 x,y를 n등분하고 각각 model로 fitting한 후
    전체 성능을 계산함
    """
    score_list = []

    # iterate over split
    fit_params = {}

    for train_idx, valid_idx, test_idx in cv.split(X, y):
        model = load_pipe(cfg)  # model cv독립적으로 생성 

        # Drop Stationcode column
        usecols = [x for x in X.columns if x not in cfg.drop_cols]
        X = X[usecols]
        assert X.dtypes.value_counts().size == 1

        y = y.astype(int)

        # model이 nested pipeline이므로 steps[-1][0]은 estimator name string임
        #  steps[-1][1]은 model object
        # 지금 추가하는 파라미터는 계산이 이뤄져야하는 값을 넣는거라
        #  config에서 지정 못한거임
                # validation set 지정
        if cfg.model_name == 'xgb':
            obs = y[test_idx]
        else:
            obs = y.iloc[test_idx]

        if sum(obs) == 0 :
            print(f'!!! there is not fog label in this split')
            continue

        # class weight 지정
        if cfg.model_name in ['cb', 'rf', 'lgb']:
            fit_params[f"{model.steps[-1][0]}__sample_weight"] = sample_weight[train_idx]
        else:
            pass # xgb 는 scale_pos_weight가 defalut로 적용됨

        # validation set 지정
        if cfg.model_name == 'xgb':
            X2 = X.values
            y2 = y.values
            fit_params[f"{model.steps[-1][0]}__eval_set"] = [(X2[valid_idx], y2[valid_idx])]
            model.fit(X2[train_idx], y2[train_idx], **fit_params)
          
            # pred3 = model.predict_proba(X2[test_idx])  # 자동으로 iteration_range가 적용되어 bestmodel이 사용된 것으로 보임
            # pred4 = model.predict_proba(X2[test_idx], iteration_range = (0,model[1].best_iteration+1))
            # pred == pred2, pred2 != pred3 ,pred 3 == pred4 (noramlization 거친 prediction)
            # pred_params = {}
            # pred_params[f"{model.steps[-1][0]}__iteration_range"] = (0,model[1].best_iteration+1)
            # pred2_2 = model[1].predict_proba(X2[test_idx], **pred_params)  # predict_proba() got an unexpected keyword argument 'xgbclassifier__iteration_range'
            pred = model.predict(X2[test_idx])
        else:
            fit_params[f"{model.steps[-1][0]}__eval_set"] = (X.iloc[valid_idx], y.iloc[valid_idx]) 
            model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_params)
            
            pred = model.predict(X.iloc[test_idx])
        
        assert pred.ndim == 1
        selection_mask = np.logical_not(np.isnan(obs))
        score = calc_metrics(obs[selection_mask], pred[selection_mask])  # test set 결과 뽑는 것임
        score_list.append(score)

        """
        #FIXME: save best config, 
        study = optuna.load_study(
                cfg.optuna_config.study_name, storage=cfg.optuna_config.storage)

        trial = study.best_trial
        best_model_path = trial.user_attrs[f"best_model_path_cv{i}"]
        # hparams = OmegaConf.load(
        #     os.path.dirname(os.path.dirname(best_model_path)) + "/" + "hparams.yaml")

        # load model
        model: LightningModule = get_module(cfg.model_name).load_from_checkpoint(
            study.best_trial.user_attrs[f"best_model_path_cv{i}"])
        
        base_pth, model_name = best_model_path.split('tb_logger')[0], best_model_path.split('tb_logger')[-1]
        #copy
        from shutil import copy
        copy(os.path.dirname(os.path.dirname(best_model_path)) + "/" + "hparams.yaml",
                f'{base_pth}/best_hparams_cv{i}.yaml')
        copy(study.best_trial.user_attrs[f"best_model_path_cv{i}"],
                f'{base_pth}/best_model_cv{i}.ckpt')
        """

    # 각각의 성능을 계산하고 평균하는 것이 아닌 전체 결과를 합쳐서 성능을 평가하는 방식으로 K-FOLD함.
    # concatenate cross validation result (1d numpy array) 
    metrics = ["ACC", "PAG", "POD", "F1"]
    arr = np.array([score_list[i][metric] for metric in metrics for i in range(len(score_list)) ])
    macro_score = np.mean(arr.reshape(-1,len(score_list)), axis=1).tolist()
    # TP, TN, FP, FN 반환하려면 np.sum으로 바꾸고 나누기 33
    return macro_score


# get estimator
def get_estimator(model_name, model_params: Optional[Dict] = None):
    if model_params is None:
        model_params = {}

    if model_name == "rf":    
        return lightgbm.LGBMClassifier(
            #  gbdt, traditional Gradient Boosting Decision Tree.
            #  dart, Dropouts meet Multiple Additive Regression Trees.
            #  goss, Gradient-based One-Side Sampling.
            #  rf, Random Forest.
            **model_params,
        )

    elif model_name == "lgb":
        return lightgbm.LGBMClassifier(
            **model_params )

    elif model_name == "cb":
        return catboost.CatBoostClassifier(
            **model_params, )

    elif model_name == "xgb":
        return xgb.XGBClassifier(
            **model_params,
        )


# load scikit-learn pipeline
def load_pipe(cfg: DictConfig):
    base_model = get_estimator(cfg.model_name, cfg.model_params)
    pipe = make_pipeline(
        StandardScaler(),
        base_model,
    )
    return pipe


def _main(cfg: Union[DictConfig, OmegaConf]):
    # step2에서는 hyper param tuning하고 각 trial별 성능만 기록함.
    # cfg.state 종류 : train, refit ,test

    if cfg.stage == 'train':
        X, y, cv = load_data(cfg)
        sample_weight = np.ones_like(y)
        mask = (y == 1)
        sample_weight[~mask] = 1 / cfg.pos_label_weights  # give less weight in y==0 samples
        # for station_code, pos_label_weight in cfg.pos_label_weights.items():
        #     mask = np.logical_and(y == 1, X.station_code == station_code)
        #     sample_weight[mask] = pos_label_weight
        
        metrics = custom_cross_val_predict(X, y, cv, sample_weight, cfg)
        OmegaConf.save(
            OmegaConf.create(metrics),
            'metrics.yaml'
        )
        return metrics[-1] # metrics['F1']


@hydra.main(config_path="../conf", config_name="config_FC10.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    
    # cfg.model_params.max_depth에 따라 변화해야하는 값인데 
    # config.yaml 파일에서는 변수형으로 정수값을 줄 수 없어서 여기서 추가함
    
    # OmegaConf.set_struct(cfg, True)
    # if cfg.model_name in ['lgb', 'rf']:
    #     with open_dict(cfg):
    #         cfg.model_params.num_leaves = 2**cfg.model_params.max_depth-1

    # elif cfg.model_name in ['xgb']:  # xgb는 이렇게 하면 val_loss가 안떨어지는 거 보고 작게 tuning하기로 함
    #     with open_dict(cfg):
    #         cfg.model_params.max_leaves = 2**cfg.model_params.max_depth-1
        
    return _main(cfg)
    
if __name__ == "__main__":
    main()

