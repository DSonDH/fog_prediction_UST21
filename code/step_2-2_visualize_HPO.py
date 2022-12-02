from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.collections import LineCollection
from omegaconf import DictConfig, OmegaConf

db_name = 'study_TPESOpt_MV.db'
n_trial = 300
exp_name = 'TPESOpt_MV'
for port in ['SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006', 
             'SF_0007', 'SF_0008', 'SF_0009', 'SF_0010', 'SF_0011']:
    
    for pred_hour in [1, 3, 6]:
        for model in ['cb', 'lgb', 'rf', 'xgb']:
            loaded_study = optuna.load_study(
                                    study_name=f"{port}_{pred_hour}_{model}", 
                                    storage=f"sqlite:///{db_name}")
            record = loaded_study.trials_dataframe()[-1*n_trial:]

            record_model = record[
                    record['user_attrs_model_cfg@_global_'] == model]
            f1 = record_model['value'].to_numpy()
            
            # read config file
            log_prefix = f'./data/log_{exp_name}/2.1/{port}/'\
                         f'{pred_hour}/{model}/multirun'
            config_files = [config_file for config_file in 
                                Path(log_prefix).glob('**/.hydra/config.yaml')]
            config_files = [item for item in config_files 
                                                if 'multirun' in str(item)]
            if len(config_files) < n_trial:
                continue

            pth_expDate = str(config_files[0].parent.parent.parent)

            # now record HPs of 1000 trials
            df_hps = pd.DataFrame()
            for i in range(len(config_files)):
                hps = OmegaConf.load(
                                f'{pth_expDate}/{i}/.hydra/overrides.yaml')
                hps = hps[ : hps.index('stage=train')]
                
                items = [item.split('=')[0].split('.')[-1] for item in hps]
                values = [float(item.split('=')[1]) for item in hps]
                df_hps.loc[i, items] = values
            
            # visualization    
            labelSZ = 10
            fig, axs = plt.subplots(nrows=len(df_hps.columns), figsize=(16,9))
            for i in range(len(df_hps.columns)):
                label = df_hps.columns[i]
                x = np.arange(len(df_hps.index))
                y = df_hps[label]
                
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                norm = plt.Normalize(0.5, 1.)
                lc = LineCollection(segments, array=f1, cmap='inferno', 
                                    norm=norm, linewidth=2)
                line = axs[i].add_collection(lc)
                fig.colorbar(line, ax=axs[i])
                
                axs[i].set_title(label)
                axs[i].set_ylim(y.min()*0.9, y.max()*1.1)
                axs[i].set_xlim(x.min()*0.9, x.max()*1.1)

                # --- Just drawing line without score coloring ---
                # axs[i].plot(x, y, label=label)
                # axs[i].tick_params(axis='both', labelsize=labelSZ)
                # axs[i].grid(which='major', linestyle='-', linewidth='0.5', 
                #                                             color='gray')
                # axs[i].grid(which='minor', linestyle=':', linewidth='0.1', 
                #                                             color='black')
                # axs[i].minorticks_on()

            #FIXME: f1.max()가 nan인 경우가 있다 ?! SF0009, 3H, lgb, rf일때
            plt.suptitle(f'BestTrial: {f1.argmax()}, F1: {f1.max():0.2f}')
            plt.tight_layout()
            fname = f'/home/sdh/fog-generation-ml/output/visualize_HPT/'\
                    f'{exp_name}/{port}_{pred_hour}_{model}.jpg'
            plt.savefig(fname, dpi=300)
            plt.close()

