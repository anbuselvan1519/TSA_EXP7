# Ex.No: 07 - AUTO REGRESSIVE MODEL
### Name: Anbuselvan.S
### Register No: 212223240008
### Date: 

## AIM:
To Implementat an Auto Regressive Model for yahoo stock price prediction using Python.

## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
   
## PROGRAM:

### Import necessary libraries:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
#### Read the CSV file into a DataFrame:
```py
data = pd.read_csv("/content/yahoo_stock.csv")  
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
```
#### Perform Augmented Dickey-Fuller test:
```py
result = adfuller(data['Volume']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
#### Split the data into training and testing sets:
```py
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]
```
#### Fit an AutoRegressive (AR) model with 13 lags:
```py
lag_order = 13
model = AutoReg(train_data['Volume'], lags=lag_order)
model_fit = model.fit()
```
#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF):
```py
plot_acf(data['Volume'])
plt.title('Autocorrelation Function (ACF)')
plt.show()
plot_pacf(data['Volume'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```
#### Make predictions using the AR model:
```py
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
```
#### Compare the predictions with the test data:
```py
mse = mean_squared_error(test_data['Volume'], predictions)
print('Mean Squared Error (MSE):', mse)
```
#### Plot the test data and predictions:
```py
plt.plot(test_data.index, test_data['Volume'], label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.show()
```

## OUTPUT:
### Augmented Dickey-Fuller test:
![image](https://github.com/user-attachments/assets/0e20036b-1911-4419-b2b1-c8018e6cb187)

### PACF - ACF:
![image](https://github.com/user-attachments/assets/ce38f3a9-67ee-41b6-ac90-25ed21c10dcd)
![image](https://github.com/user-attachments/assets/c261b532-ebf9-4431-8af6-44362e42e82f)

### Mean Squared Error:
![image](https://github.com/user-attachments/assets/04202b81-51f3-4751-b85b-e841198952dc)

### PREDICTION:
![image](https://github.com/user-attachments/assets/782bfb06-5d05-43d8-85b1-76f57774a985)

## RESULT:
Thus we have successfully implemented the auto regression function using python.
