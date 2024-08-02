"""
##SIMPLE LINEAR REGRESSION

Now, we define the columns on which we want to perform linear regression. Here, I have compared the performance between the correlated features by training the model based on volume,month and year along with each one of the correlated features individually.
"""

feature_list = ['Open','High','Low']

for f in feature_list:
  print("Training based on features",f,",Volume,month and year")
  X1 = df_train[[f,'Volume','year','month']]
  y1 = df_train['Adj Close']
  X2 = df_test[[f,'Volume','year','month']]
  y2 = df_test['Adj Close']
  # build a model
  st = time.time()
  linear_regressor = linear_model.LinearRegression()
  linear_regressor.fit(X1, y1)
  et=time.time()
  print("Time to fit: ",(et-st),"s")
  y_pred=linear_regressor.predict(X2)
  # Model evaluation
  mse = mean_squared_error(y2, y_pred)
  r2 = r2_score(y2, y_pred)
  print("Mean Squared Error:", mse)
  print("R-squared:", r2)

"""From the above results we can see that when we train based on the features low, volume, month and year, we seem to get the best result. This seems to be the case after a few re-runs.
"""