from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def entrenar_sarima(train, p=3,d=1,q=2,P=1,D=0,Q=0,m=14):
    model = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,m),
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    return model_fit

def forecast(model_fit, steps):
    forecast_res = model_fit.get_forecast(steps=steps)
    return forecast_res.predicted_mean, forecast_res.conf_int()

def calcular_metricas(test, pred):
    test_weekly = test.resample('W-SUN').sum()
    pred_weekly = pred.resample('W-SUN').sum()
    rmse = mean_squared_error(test_weekly.values, pred_weekly.values)**0.5
    mask_nonzero = test_weekly != 0
    mape = (abs((test_weekly[mask_nonzero] - pred_weekly[mask_nonzero]) / test_weekly[mask_nonzero])).mean() * 100
    return rmse, mape, test_weekly, pred_weekly
