import shutil
from pathlib import Path
from typing import *

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # eval_metric 하나만 가지고 Hyperparameter만 알 수 있는 상태이므로 아래 코드는 돌아가지 않음.
    # 그렇다면 최종 성능을 구하기 위해서는 refit_test돌려야함

    # given various train output configures,
    config_files = [
        config_file for config_file in (Path(get_original_cwd()) / cfg.log_prefix).glob('**/.hydra/config.yaml')
        if OmegaConf.load(config_file).api_version == cfg.api_version and OmegaConf.load(config_file).stage == 'train'
    ]  # ??? debug로 해야 config_files 찾아진다 ???

    metric_files = [
        config_file.parent.parent.parent / 'metrics.yaml' for config_file in config_files
    ]  
    # config_files[0].parent.parent.parent/'optimization_results.yaml'

    # select best configure file which metric was sorted last(best score)
    metric_files = [x for x in metric_files if x.exists()]
    best_metric_file = sorted(
                                metric_files,
                                key = lambda metric_file: OmegaConf.load(metric_file)[-1] 
                       )[-1]

    best_config_file = best_metric_file.parent / '.hydra' / 'config.yaml'

    # cv result
    dest = Path(get_original_cwd()) / cfg.catalogue.best_params
    dest.mkdir(parents=True, exist_ok=True)
    base = r'/home/sdh/fog-generation-ml/'
    
    shutil.copyfile(str(best_config_file), f'{base}/{cfg.catalogue.best_params}/config.yaml')
    shutil.copyfile(str(best_metric_file), f'{base}/{cfg.catalogue.best_params}/metrics.yaml')
    # metrics = ["ACC", "CSI", "PAG", "POD", "F1"]
    return 0

if __name__ == "__main__":
    main()
