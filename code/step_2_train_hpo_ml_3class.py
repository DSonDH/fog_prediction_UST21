import warnings
from typing import *

import catboost
import hydra
import lightgbm
import numpy as np
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from step_1_pre_train_ml_3class import load_data
from utils import calc_metrics

warnings.simplefilter(action='ignore', category=FutureWarning)


# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(X, y, cv, label0_weight, sample_weight, smpl_freq, cfg):
    """
    model, x, y, cross validator, fit_params가 주어지면
    cv로 x,y를 n등분하고 각각 model로 fitting한 후
    전체 성능 계산
    """

    fit_params = {}

    # Drop Stationcode column
    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]
    y = y.astype(int)
    assert X.dtypes.value_counts().size == 1

    cv_idx = 0
    for train_idx, valid_idx, test_idx in cv.split(X, y):
        cv_idx += 1
        if cv_idx > 1:
            break
        
        # loop안에서 정의해야 model cv독립적으로 생성됨
        model = load_pipe(cfg, label0_weight, smpl_freq)  

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
        try:  # LGB에 is_unbalance같은 option주고 학습하면 가끔 error뜸.
            model.fit(X_tr, y_tr, **fit_params)
        except:
            print(f'{cfg.model_name} crashed !!!!')
            print(f'{cfg.model_name} try one more time !!!!')
            model.fit(X_tr, y_tr, **fit_params)
        
        pred = model.predict(X_te)
        
        # get test score
        selection_mask = np.logical_not(np.isnan(y_te))
        score = calc_metrics(y_te[selection_mask], pred[selection_mask], 
                            binary= False)

        # check cb feature importance
        # ft_importance = model.steps[-1][-1].get_feature_importance()
        # impt_mask = ft_importance > 0
        # items = X.columns[impt_mask]
        # values = np.round(ft_importance[impt_mask]/100, 4)
        # if len(items) > 10:
        #     ft_dict = {items[i]:values[i] for i in range(len(values))}
        #     ft_dict = {k: v for k, v in sorted(ft_dict.items(), key=lambda item: item[1], reverse=True)}
        #     ft_dict2 = {list(ft_dict.keys())[i]: list(ft_dict.values())[i] for i in range(10)}
        #     items = list(ft_dict2.keys())
        #     values = list(ft_dict2.values())

        # temp code for parameter tuning
        # from sklearn.metrics import multilabel_confusion_matrix
        # print(f"macro_scores : [{score['macro_PAG']*100:.1f}, {score['macro_POD']*100:.1f}, "\
        #       f"{score['macro_F1']*100:.1f}]")
        # mcm = multilabel_confusion_matrix(y_te, pred)
        # for i in range(3):
        #     TN, FP, FN, TP = mcm[i].flatten()
        #     label = i
        #     acc = np.round((TN + TP) / (TN + FP + FN + TP) * 100, 2)
        #     pag = np.round((TP) / (TP + FP) * 100, 2)
        #     pod = np.round((TP) / (TP + FN) * 100, 2)
        #     f1 = np.round((2 * pag * pod) / (pag + pod), 2)
        #     print(f"{cfg.pred_hour}H_label{label} : [{pag}, {pod}, {f1}]")
        
        return score
        # import optuna        
        # loaded_study = optuna.load_study(
        #             study_name=f"SF_0003_6_cb",
        #             storage=f"sqlite:///study_3class.db")
        # best_record = loaded_study.best_trial
        # best_f1 = best_record._values[0]
        # best_modelParams = best_record._params
        


# get estimator
def get_estimator(weight, sample_freq, model_name, model_params: Optional[Dict] = None):
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
def load_pipe(cfg: DictConfig, weight, smpl_freq):
    base_model = get_estimator(weight, smpl_freq, cfg.model_name, cfg.model_params)
    pipe = make_pipeline(
        StandardScaler(),
        base_model,)
    return pipe


def _main(cfg: Union[DictConfig, OmegaConf]):
    # step2에서는 hyper param tuning하고 각 trial별 성능만 기록함.
    # cfg.state 종류 : train, refit ,test

    if cfg.stage == 'train':
        X, y, cv = load_data(cfg)

        sample_weight = np.ones_like(y)
        mask0 = (y == 0)
        mask1 = (y == 1)
        mask2 = (y == 2)
        
        smpl_freq = [1, (sum(mask0))/(sum(mask1) + 1), (sum(mask0))/(sum(mask2) + 1)]
        label0_weight = (sum(mask0))/(sum(mask2) + 1) * cfg.pos_label_ratio
        # label1_weight = (sum(mask1))/(sum(mask2) + 1) * cfg.pos_label_ratio
        # label1도 잘 맞히고자 하는 거니깐 보통 시정만 가중치 낮춰보자
        
        sample_weight[mask0] = 1 / label0_weight  # give less weight
        # sample_weight[mask1] = 1 / label1_weight  # give less weight

        metrics = custom_cross_val_predict(X, y, cv, label0_weight, sample_weight, smpl_freq, cfg)
        OmegaConf.save(
            OmegaConf.create(metrics),
            'metrics.yaml'
        )
        return metrics['macro_F1']


@hydra.main(config_path="../conf", config_name="config_3class.yaml", version_base=None)
def main(cfg: DictConfig) -> None:

    return _main(cfg)
    
if __name__ == "__main__":
    main()
