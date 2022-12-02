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
def custom_cross_val_predict(X, y, cv, label1_weight, sample_weight, cfg) -> List:
    """
    model, x, y, cross validator, fit_params가 주어지면
    cv로 x,y를 n등분하고 각각 model로 fitting한 후
    전체 성능 계산
    """
    score_list = []
    fit_params = {}

    # Drop Stationcode column
    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]
    y = y.astype(int)
    assert X.dtypes.value_counts().size == 1

    for train_idx, valid_idx, test_idx in cv.split(X, y):
        # loop안에서 정의해야 model cv독립적으로 생성됨
        model = load_pipe(cfg, label1_weight)

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

        fit_params[f"{model.steps[-1][0]}__sample_weight"] = \
                                                        sample_weight[train_idx]
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
        score = calc_metrics(y_te[selection_mask], pred[selection_mask])
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
    # TP, TN, FP, FN 반환하려면 np.sum으로 바꾸고 나누기
    return macro_score


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
            **model_params )

    elif model_name == "cb":
        return catboost.CatBoostClassifier(
            # class_weights = [1, weight],
            **model_params, )

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
        
        label1_weight = (len(y)-sum(y))/(sum(y) + 1) * cfg.pos_label_ratio
        sample_weight[~mask] = 1 / label1_weight  # give less weight in y==0 samples

        metrics = custom_cross_val_predict(X, y, cv, label1_weight, sample_weight, cfg)
        OmegaConf.save(
            OmegaConf.create(metrics),
            'metrics.yaml'
        )
        return metrics[-1] # metrics['F1']


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
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

