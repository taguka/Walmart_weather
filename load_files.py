import pandas as pd
import numpy as np

weather = pd.read_csv(('C:/Kaggle/Walmart_weather/data/weather.csv'), na_values=["M", "-", "*","  T"])
key = pd.read_csv('C:/Kaggle/Walmart_weather/data/key.csv')
training = pd.read_csv('C:/Kaggle/Walmart_weather/data/train.csv')
testing = pd.read_csv('C:/Kaggle/Walmart_weather/data/test.csv')
print('Tot weather: %i, key: %i, train: %i, test: %i' %(weather.shape[0],key.shape[0],training.shape[0],testing.shape[0]))

codesum_columns = set(' '.join(set(weather["codesum"])).strip().split())
codesum = pd.DataFrame(index=weather.index, columns=codesum_columns)
for column in codesum.columns:
    for i in range(len(weather.index)):
        if column in weather["codesum"][i]:
            codesum[column][i] = 1
weather = weather.drop('codesum', 1)
weather = weather.join(codesum.fillna(0))                   
weather.to_csv('C:/Kaggle/Walmart_weather/data/weather_modified.csv')             