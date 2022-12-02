from pathlib import Path
from typing import *

import hydra
import joblib
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from step_1_pre_train_ml import load_data
from step_2_train_hpo_ml import load_pipe
# from tune_sklearn import TuneSearchCV
from utils import calc_metrics


# evaluation function
def evaluation_fn(model, X, y):
    test_data = joblib.load(cfg.data.interim.test_path)
    X = test_data['X']
    y = test_data['y']
    pred = model.predict(X)
    # ignore na
    selection_mask = np.logical_not(np.isnan(y))
    model_path = Path(cfg.root) / "model" / cfg.station
    joblib.dump(model, model_path)
    # calc metrics
    metrics = calc_metrics(y[selection_mask], pred[selection_mask])
    payload = {}
    payload["pred"] = pred
    payload["test"] = y
    payload["metrics"] = metrics
    # return payload
    return payload


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    # given various train output configures,
    config_files = [
        config_file for config_file in (Path(get_original_cwd()) / cfg.log_prefix).glob('**/.hydra/config.yaml')
        if OmegaConf.load(config_file).api_version == cfg.api_version and OmegaConf.load(config_file).stage == 'train'
    ]

    metric_files = [
        config_file.parent.parent / 'metrics.yml' for config_file in config_files  # FIXME yml to yaml
    ]

    # select best configure file which metric was sorted last(best score)
    metric_files = [x for x in metric_files if x.exists()]
    best_metric_file = sorted(
        metric_files,
        key = lambda metric_file: OmegaConf.load(metric_file)[-1] 
    )[-1]
    # posixpath 설명
    # https://ryanking13.github.io/2018/05/22/pathlib.html

    best_config_file = best_metric_file.parent / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(best_config_file)
    cfg.stage = 'refit'


    # refit
    X, y = load_data(cfg)
    sample_weight = np.ones_like(y)
    mask = (y == 1)
    sample_weight[~mask] = 1 / cfg.pos_label_weights

    # load pipe to refit
    model = load_pipe(cfg)
    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]

    assert X.dtypes.value_counts().size == 1
    y = y.astype(int)
    
    fit_params = {}
    fit_params[f"{model.steps[-1][0]}__sample_weight"] = sample_weight
    
    # fit_params[f"{model.steps[-1][0]}__{string}"] = cfg.pos_label_weights

    # do refit
    model.fit(X, y, **fit_params,)

    # save refitted model
    dest = Path(get_original_cwd()) / cfg.catalogue.model
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        model,
        dest            
    )

    #%% test
    cfg.stage = 'test'

    model = joblib.load(
        Path(get_original_cwd()) / cfg.catalogue.model)
    X_test, y_test = load_data(cfg)
    pred = model.predict(X_test)
    
    metrics = calc_metrics(y_test, pred)
    metrics['stage'] = 'test'
    
    # save test result
    dest = Path(get_original_cwd()) / cfg.catalogue.test_result
    dest.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(metrics),
        dest
    )

if __name__ == "__main__":
    main()
