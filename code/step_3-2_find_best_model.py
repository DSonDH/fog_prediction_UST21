import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# SF_0001: 부산항(북항)
# SF_0002: 부산항(신항)
# SF_0003: 인천항
# SF_0004: 평택당진항
# SF_0005: 군산항
# SF_0006: 대산항
# SF_0007: 목포항
# SF_0008: 여수항
# SF_0009: 해운대
# SF_0010 : 울산항
# SF_0011: 포항항

port_num_dict = {
    'Busan':'SF_0001', 'NBusan':'SF_0002', 'Incheon':'SF_0003', 'PTDJ':'SF_0004', 'Gunsan':'SF_0005', 
    'Daesan':'SF_0006', 'Mokpo':'SF_0007', 'Yeosu':'SF_0008', 'Haeundae':'SF_0009', 'Ulsan':'SF_0010', 'Pohang':'SF_0011'}

df = pd.DataFrame(columns=["ACC", "CSI", "PAG", "POD", "F1", 'model'], index=['1', '3', '6'])

with pd.ExcelWriter(f'./output/best_params_allModel/2.1/all_port_bestScore.xlsx') as excel_writer:
    for port in ['Daesan', 'Mokpo']:
        for pred_hour in ['1', '3', '6']:
            
            best_score = 0
            for model in ['cb', 'lgb', 'rf', 'xgb']:
                path = f'output/best_params_allModel/2.1/{port_num_dict[port]}/{pred_hour}/{model}'
                metrics = OmegaConf.load(f'{path}/metrics.yaml')  #["ACC", "CSI", "PAG", "POD", "F1"]
                metrics = np.round(metrics,4)*100

                if metrics[-1] > best_score:
                    best_score = metrics[-1]
                    best_metrics = metrics
            df.loc[pred_hour,:] = list(best_metrics)+[model]
        
        df.to_excel(excel_writer, sheet_name =port, index_label=port, header=True)
        print(df)


            