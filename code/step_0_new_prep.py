import os

import joblib
import numpy as np
import pandas as pd


def preprocess(df):
    station_now = df.station.unique()[0]
    if station_now in vaisala_stations:
        pois_start = '2017-01-01'
    else:
        pois_start = '2018-01-01'

    df = df.loc[pois_start:, :]
    radian_wd = np.deg2rad(df.wd)
    df['u'] = df.ws * np.sin(radian_wd)
    df['v'] = df.ws * np.cos(radian_wd)
    df['vis'] = df['vis20'].clip(20, 3000)

    aggregator = {
        'u': 'mean',
        'v': 'mean',
        'temp': 'mean',
        'sst': 'mean',
        'qff': 'mean',
        'rh': 'mean',
        'vis': 'mean',
    }

    df['std_ws'] = df['ws'].rolling('10T').std()
    df['std_ASTD'] = (df['temp'] - df['sst']).rolling('10T').std()
    df['std_rh'] = df['rh'].rolling('10T').std()
    # 최근 60분 1분간격 해무 빈도
    df['fog_count'] = df.vis.le(1000).rolling('60T').sum()
    data_cols = ['u', 'v', 'temp', 'sst', 'qff', 'rh', 'vis']

    std_statistics = df.loc[:, ['std_ws', 'std_ASTD', 'std_rh']]

    df = df[data_cols].rolling('10T').agg(aggregator).join(df.fog_count)
    # resample로 하면 라벨이 (] 왼쪽 고정이라 roling으로 함
    df = df[df.index.minute % 10 == 0]  
    

    df['station'] = ports_code_dict[station_now]
    stations = df.station

    covariates = [x for x in data_cols if x != 'vis']
    # 일단 시정빼고 1시간 보간
    df[covariates] = df[covariates].interpolate(method='linear', limit=6)
    df['vis'] = df['vis'].fillna(3000)  # 시정은 선형보간하기 애매해서 최빈값 보간
    df['T'] = df.temp + 273.15  # 섭씨 -> 절대
    df['ASTD'] = df.temp - df.sst
    df['Td'] = df.temp - ((100 - df.rh) / 5) * np.sqrt(df['T'] / 300) \
                        - 0.00135 * (df.rh - 84) ** 2 + 0.35
    df['temp-Td'] = df.temp - df.Td
    df['sst-Td'] = df.sst - df['Td']

    assert (df.index[1:] - df.index[:-1]).nunique() == 1
    assert (df.index[1:] - df.index[:-1]).value_counts().idxmax() == \
                                            pd.Timedelta(minutes=10)

    label = df.vis.copy()

    label[:] = np.select(
        [df.fog_count >= 50, df.fog_count < 50],
        [1, 0]
    )
    y = pd.DataFrame({
        'y_1': label.shift(-1 * 6),
        'y_3': label.shift(-3 * 6),
        'y_6': label.shift(-6 * 6)
    })
    

    y.set_index = df.index

    df = df.loc[:, ['u', 'v', 'temp', 'sst', 'qff', 'rh', 'vis', 'ASTD', 'Td', 
                    'temp-Td', 'sst-Td', 'fog_count']]
    x = pd.concat(
        [df.shift(lag).add_prefix(f'lag_{lag:02d}_') for lag in range(0, 13)], 
                axis=1)  # 0은 현재고 12는 2시간 전

    target_time = x.index
    timestamp_as_sec = target_time.view(np.int64) // 1e+9  #
    assert timestamp_as_sec[1] - timestamp_as_sec[0] == 600
    hour = 60 * 60
    day = 24 * hour
    year = (365.2425) * day
    del hour

    day_sin = np.sin(timestamp_as_sec * (2 * np.pi / day))
    day_cos = np.cos(timestamp_as_sec * (2 * np.pi / day))
    year_sin = np.sin(timestamp_as_sec * (2 * np.pi / year))
    year_cos = np.cos(timestamp_as_sec * (2 * np.pi / year))

    y = y.assign(
        time_day_sin=day_sin,
        time_day_cos=day_cos,
        time_year_sin=year_sin,
        time_year_cos=year_cos,
    )

    station_ohe = pd.get_dummies(stations.astype(pd.CategoricalDtype(
                                                    categories=ports_code)))
    y = y.join(std_statistics).join(station_ohe).join(stations)

    x.index.name = 'datetime'
    y.index.name = 'datetime'

    x.columns = columns_x
    y.columns = columns_y

    joblib.dump(
        {
            'x': x,
            'y': y
        },
        save_prep_path+f'/{ports_code_dict[station_now]}.pkl'
    )


if __name__ == '__main__':
    old_path = './raw_old'
    new_path = './raw_new'
    # Multi column index 구성하기 귀찮아서 전처리 된 파일(ref file) column을 가져옴.
    ref_path = './ref'
    save_raw_path = './save_raw'
    save_prep_path = './save_preprocessed'
    ports_code_dict = {
        '군산항': 'SF_0005',
        '대산항': 'SF_0006',
        '목포항': 'SF_0007',
        '부산항': 'SF_0001',
        '부산항(부산신항)': 'SF_0002',
        '여수항': 'SF_0008',
        '울산항': 'SF_0010',
        '인천항': 'SF_0003',
        '평택당진항': 'SF_0004',
        '포항항': 'SF_0011',
        '해운대': 'SF_0009'
    }
    ports_code = [
        f'SF_00{i:02}' for i in range(1, 12)
    ]
    vaisala_stations = ['인천항', '평택당진항', '부산항(부산신항)', '부산항']
    dtypes = {
        '항': str,
        '풍향(수치)': float,
        '풍속(m/s)': float,
        '기온(℃)': float,
        '수온(℃)': float,
        '기압(hPa)': float,
        '습도(%)': float,
        '시정(3km)': float,
        '시정(20km)': float,
        'source': str
    }
    renamer = {
        '항': 'station',
        '풍향(수치)': 'wd',
        '풍속(m/s)': 'ws',
        '기온(℃)': 'temp',
        '수온(℃)': 'sst',
        '기압(hPa)': 'qff',
        '습도(%)': 'rh',
        '강수량(mm)': 'precipitation',
        '시정(3km)': 'vis3',
        '시정(20km)': 'vis20'
    }
    vaisala_stations = ['인천항', '평택당진항', '부산항신항', '부산항']
    new_port_names = [
        '군산항', '대산항', '목포항', '부산항', '부산항(부산신항)', '여수항', 
        '울산항', '인천항', '평택당진항', '포항항', '해운대'
    ]
    old_port_names = [
        '군산항', '대산항', '목포항', '부산항', '부산항신항', '여수광양항', '울산항',
        '인천항', '평택당진항', '포항항', '해운대'
    ]

    new_data = pd.read_csv(
        new_path + '/input_data.csv', encoding='euc-kr',
        parse_dates=['시간'], index_col=['시간'], dtype=dtypes
    )
    new_data = new_data.rename(columns=renamer)

    ref_files = os.listdir(ref_path)
    ref_data = joblib.load(
        ref_path + '/' + ref_files[0]
    )
    columns_x = ref_data['x'].columns
    columns_y = ref_data['y'].columns

    for new_port_name, old_port_name in zip(new_port_names, old_port_names):
        old_data = pd.read_csv(
            old_path + f'/{old_port_name}.csv.zip',
            parse_dates=['시간'], index_col=['시간'], dtype=dtypes
        )
        old_data = old_data.rename(columns=renamer)
        con_data = pd.concat([
            old_data, new_data.loc[new_data.station == new_port_name, :]
        ])
        con_data = con_data.sort_index()
        con_data.to_csv(
            save_raw_path + f'/{new_port_name}.csv', encoding='euc-kr'
        )
        con_data.station = new_port_name
        preprocess(con_data)
