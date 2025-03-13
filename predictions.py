#%%
# imports
from tkinter import*
from matplotlib import pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from datetime import date, timedelta

# date + time
Start = date.today() - timedelta(1825)
Start.strftime('%Y-%m-%d')
End = date.today() + timedelta(2)
End.strftime('%Y-%m-%d')


ticker = input('Enter Stock / Index / Crytpo Ticker:  ').upper()
company_name = str(yf.Ticker(ticker).info['longName'])

# fetch stock from yfinance
def full_price(ticker):
    Asset_full = pd.DataFrame(yf.download(ticker, start=Start, end=End, progress=False)[['Open', 'Volume', 'Close']])
    Asset_full['MA50'] = Asset_full['Close'].rolling(window=50).mean()
    Asset_full['MA200'] = Asset_full['Close'].rolling(window=200).mean()
    Asset_full['RSI'] = 100 - (100 / (1 + Asset_full['Close'].pct_change().rolling(14).mean() /
                                         Asset_full['Close'].pct_change().rolling(14).std()))
    Asset_full['Lag1'] = Asset_full['Close'].shift(1)
    Asset_full['Lag2'] = Asset_full['Close'].shift(2)
    Asset_full.dropna(inplace=True)
    return Asset_full

full_data = full_price(ticker)

# features
scaler = MinMaxScaler()
features = ['Open', 'Volume', 'MA50', 'MA200', 'RSI', 'Lag1', 'Lag2']
full_data[features] = scaler.fit_transform(full_data[features])

# data split
tscv = TimeSeriesSplit(n_splits=5)

# target
target = 'Close'

# make + train model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

# train model with the split data
for train_index, test_index in tscv.split(full_data):
    train, test = full_data.iloc[train_index], full_data.iloc[test_index]
    model.fit(train[features], train[target])
    prediction = model.predict(test[features])

# measure accuracy
mae = mean_absolute_error(test[target], prediction)
rmse = np.sqrt(mean_squared_error(test[target], prediction))
mape = np.mean(np.abs((test[target].values - prediction) / test[target].values)) * 100
accuracy = model.score(test[features], test[target])

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'Accuracy: {int(accuracy*100)}%')

# making graph
plt.plot(test.index, test[target], label='Actual Price')
plt.plot(test.index, prediction, label='Predicted Price', color='red')
plt.title(f"Closing Value Predictions for: {company_name}")
plt.legend()
plt.show()
