The dataset on which we work provides historical data from 2012 to 2022 of TESLA INC. stock (TSLA) in USD. The data is available at a daily level. Here, I try to regress and predict the adjusted closing price of the stock.
Here, there is a flaw that has not been addressed. All these methods are uded to predict independent and identically distributed random values, however that is not the case in the case of time series data. Rather, methods like Autoregression and Moving Averages model should be utilised. The only properly used method is the Long Short Term neural network in the project. However, the main motive of the project was to get used and accustomed to the various different regression methods available and that has been successfully met.

Dataset: TESLA Stock Data. (2022, March 25). Kaggle. https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021/code

It has the following features:

Date: The date of the trading day.

Open: The opening price of the stock on that trading day.

High: The highest price the stock reached during that trading day.

Low: The lowest price the stock reached during that trading day.

Close: The closing price of the stock on that trading day.

Adj Close: The adjusted closing price, which accounts for any corporate actions such as stock splits, dividends, etc.

Volume: The trading volume, i.e., the number of shares traded during that trading day.

Aim: We will try and predict the adjusted close prices using all of the relevant features using different methods of regression. Then, we would perform a comparative analysis of each method and provide a final verdict as to which method(s) suits this dataset the best to predict future adjusted close prices.

References:

Sksujanislam. (2023, October 13). MULTIVARIATE TIME SERIES FORECASTING USING LSTM - Sksujanislam - Medium. Medium. https://medium.com/@786sksujanislam786/multivariate-time-series-forecasting-using-lstm-4f8a9d32a509

Zahraaalaatageldein. (2024, January 29). 🥏Deep Analysis with NNs [ups&downs]. https://www.kaggle.com/code/zahraaalaatageldein/deep-analysis-with-nns-ups-downs

Omarrezk. (2023, October 26). Tesla Stock EDA & Prediction. https://www.kaggle.com/code/omarrezk/tesla-stock-eda-prediction
