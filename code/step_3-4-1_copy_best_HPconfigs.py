import os
from pathlib import Path
from shutil import copy
from typing import *

import hydra
import optuna
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def _main(cfg: Union[DictConfig, OmegaConf]):
    db_name = 'study_221017.db'

    for station in ['SF_0001', 'SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 'SF_0007', 'SF_0008',  'SF_0009', 'SF_0010', 'SF_0011']:
        for pred_hour in [1, 3, 6]:

            # loaded_study = optuna.load_study(study_name=f"{station}_{pred_hour}", storage=f"sqlite:///{db_name}")
            # best_record = loaded_study.best_trial
            # best_model = best_record.user_attrs['model_cfg@_global_']
            # best_modelPrams = best_record._params
            # best_f1 = best_record._values[0]

            for model in ['cb', 'lgb', 'rf', 'xgb']:
                # save best score for all models
                log_prefix = '/'.join(cfg.log_prefix.split('/')[:4]) + f'/{station}/{pred_hour}/{model}'
                path_gen = (Path(get_original_cwd()) / log_prefix).glob('**/.hydra/config.yaml')
                one_config = path_gen.__next__()
                best_config = one_config.parent.parent.parent/'optimization_results.yaml'

                if not best_config.exists():
                    print(f'!!!! There is no {best_config}')
                    continue

                dst = f'./output/best_configs/best_{station}_{pred_hour}_{model}.yaml'
                copy(best_config, dst)

    

@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()
