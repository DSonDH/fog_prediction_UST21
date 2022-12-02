from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Detrender

import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt


"""
code instruction from 
blog:
https://towardsdatascience.com/multi-step-time-series-forecasting-with-arima-lightgbm-and-prophet-cc9e3f95dfb0

colab:
https://colab.research.google.com/drive/1Z4zNI_bVXoFQBsCHUtxBDCBno6yhXceB?usp=sharing#scrollTo=EggD0D7FmNly
"""


def create_forecaster():
    
    # creating forecaster with LightGBM
    regressor = lgb.LGBMRegressor()
    forecaster = make_reduction(regressor, window_length=5, strategy="recursive")
    
    return forecaster


def grid_serch_forecaster(train, test, forecaster, param_grid):

    # Grid search on window_length
    cv = ExpandingWindowSplitter(initial_window=int(len(train) * 0.7))
    gscv = ForecastingGridSearchCV(
            forecaster, strategy="refit", cv=cv, param_grid=param_grid
           )
    gscv.fit(train)
    print(f"best params: {gscv.best_params_}")
    
    # forecasting
    fh=np.arange(len(test))+1
    y_pred = gscv.predict(fh=fh)
    mae, mape = plot_forecast(train, test, y_pred)

    return mae, mape


def plot_forecast(series_train, series_test, forecast, forecast_int=None):

    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MAPE: {mape:.3f}", size=18)
    series_train.plot(label="train", color="b")
    series_test.plot(label="test", color="g")
    forecast.index = series_test.index
    forecast.plot(label="forecast", color="r")
    if forecast_int is not None:
        plt.fill_between(
            series_test.index,
            forecast_int["lower"],
            forecast_int["upper"],
            alpha=0.2,
            color="dimgray",
        )
    plt.legend(prop={"size": 16})
    plt.show()

    return mae, mape


def grid_search_forecaster(train, test, forecaster, param_grid):

    cv = ExpandingWindowSplitter(initial_window=int(len(train) * 0.7))
    gscv = ForecastingGridSearchCV(
            forecaster, strategy="refit", cv=cv, param_grid=param_grid
           )
    gscv.fit(train)
    print(f"best params: {gscv.best_params_}")
    
    # forecasting
    fh=np.arange(len(test))+1
    y_pred = gscv.predict(fh=fh)
    mae, mape = plot_forecast(train, test, y_pred)

    return mae, mape
    

# param_grid = {"window_length": [5, 10, 15, 20, 25, 30]} # parameter set to be grid searched
# forecaster = create_forecaster()
# sun_lgb_mae, sun_lgb_mape = grid_search_forecaster(
#                             sun_train, sun_test, forecaster, param_grid
#                             )

forecaster = PolynomialTrendForecaster(degree=1)
transformer = Detrender(forecaster=forecaster)
yt = transformer.fit_transform(wpi_train)

forecaster = PolynomialTrendForecaster(degree=1)
fh_ins = -np.arange(len(wpi_train))
y_pred = forecaster.fit(wpi_train).predict(fh=fh_ins)



#%%
def create_forecaster_w_detrender(degree=1):

    # creating forecaster with LightGBM
    regressor = lgb.LGBMRegressor()
    forecaster = TransformedTargetForecaster(
        [
            ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=degree))),
            (
                "forecast",
                make_reduction(regressor, window_length=5, strategy="recursive"),
            ),
        ]
    )

    return forecaster

param_grid = {"forecast__window_length": [5, 10, 15, 20, 25, 30]}
forecaster = create_forecaster_w_detrender(degree=1)
wpi_lgb_mae, wpi_lgb_mape = grid_serch_forecaster(
    wpi_train, wpi_test, forecaster, param_grid
)