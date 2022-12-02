
import joblib
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

stations = ['SF_0001', 'SF_0002', 'SF_0003', 'SF_0004', 'SF_0005', 'SF_0006',
             'SF_0007', 'SF_0008',  'SF_0009', 'SF_0010', 'SF_0011']

for station_code in stations:
    source = f'/home/sdh/fog-generation-ml/data/processed/{station_code}.pkl'
    X = joblib.load(source)["x"]    
    X = X.set_axis([f"{c1}_{c2}" for c1, c2 in X.columns], axis=1)
    y = joblib.load(source)["y"].reset_index()

    y = y.set_index("datetime")
    X.index = y.index

    
    # raw_csv = pd.read_csv(f'/home/parksh/share/project/fog-dnn/data/save_raw/{station_code}.csv')

    X_old = X[X.index < pd.to_datetime('2021-07-01')]
    X_new = X[X.index >= pd.to_datetime('2021-07-01')]
    if station_code in ['SF_0003', 'SF_0004']:
        real = pd.date_range('2017-01-01', '2022-07-01', freq='10T', closed='left')  # 236448
        real_old = pd.date_range('2017-01-01', '2021-07-01', freq='10T', closed='left')  # 183888
        real_new = pd.date_range('2021-07-01', '2022-07-01', freq='10T', closed='left')  # 52560    
    else:
        real = pd.date_range('2018-01-01', '2022-07-01', freq='10T', closed='left')  # 236448
        real_old = pd.date_range('2018-01-01', '2021-07-01', freq='10T', closed='left')  # 183888
        real_new = pd.date_range('2021-07-01', '2022-07-01', freq='10T', closed='left')  # 52560

    print(f'========  {station_code} = X_old:{len(X_old)}, ref_old:{len(real_old)}, X_new:{len(X_new)}, ref_new:{len(real_new)}')
