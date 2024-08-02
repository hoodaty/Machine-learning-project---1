"""
## LSTM

**Long Short-Term Memory (LSTM)** is a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem in traditional RNNs. It is well-suited for sequence prediction tasks, such as time series forecasting, natural language processing, and speech recognition. LSTMs have memory cells that can maintain information over long sequences, allowing them to capture dependencies and patterns in sequential data more effectively. They achieve this by using gated units, including input, forget, and output gates, which regulate the flow of information through the network, enabling it to learn long-term dependencies and handle sequences of varying lengths. This makes LSTMs particularly powerful for modeling and predicting time-dependent data where context and temporal dynamics are essential.

We derive the columns based on which we would like to train the model and the target feature (Adj Close) that we want to predict.
"""

X_ = df[['Open','High','Low','Volume','year','month']]
y_ = df[['Adj Close']]

#We scale the data in the range (0,1)
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.2)

sc = MinMaxScaler((0,1))

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

"""We define the model next."""

model_LSTM = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model_LSTM.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], 1),activation='tanh'))#input layer
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(50, return_sequences = True, activation='tanh'))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(50, return_sequences = True, activation='tanh'))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(50, activation='tanh'))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(25))
model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(1)) #output layer

model_LSTM.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])

"""**A few terms that were used above:**

**1.** `return_sequences=True` argument indicates that this layer will return sequences instead of a single output

**2.** `model_LSTM.add(Dropout(0.2))` Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of input units(neurons) to 0 during training to reduce co-dependency between neurons. Here, a dropout layer with a dropout rate of 0.2 (20%) is added after each LSTM layer.

**3.** The `activation='tanh'` argument specifies the activation function to be used, in this case, the hyperbolic tangent function.

**4.** `model_LSTM.add(Dense(1))` declares a dense layer with a single unit is added as the output layer. This layer will output a single numerical value, making it suitable for regression tasks.

We run the model with some parameters to test its performance.
"""

st = time.time()
History = model_LSTM.fit(X_train, y_train, epochs = 100, batch_size = 32)
et=time.time()
print("Time to fit: ",(et-st),"s")

"""Next we plot the $MSE$ over epochs"""

plt.plot(History.history['mean_squared_error'])
plt.title('Loss Function Over Epochs')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.show()

"""It seems to be quite clear that as the number of epochs increase, the $MSE$ seems to decreases significantly, initially, after which it almost stabilizes, having a general downward like trend.

Next we perform some predictions on the test data
"""

y_pred = model_LSTM.predict(X_test)
y_pred_copy = np.repeat(y_pred,7, axis=-1)
y_test_copy = np.repeat(y_test,7, axis=-1)

"""I perform `scaler.inverse_transform ` because I want a comparison between what the predicted values were and what are the actual values. This will help me get the values in there original unscaled form."""

y_predInverse = scaler.inverse_transform(np.reshape(y_pred_copy,(len(y_test),7)))[:,0]
y_testInverse = scaler.inverse_transform(np.reshape(y_test_copy,(len(y_test),7)))[:,0]

# Create a dataframe for predicted and original values
pred_df = pd.DataFrame({'Predicted Values': y_predInverse.flatten(), 'Original Values': y_testInverse.flatten()})
print(pred_df)

# Plot the data
plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.plot(y_predInverse[:100], color='red', label='Real Stock Price')  # Plot first 100 data points of y_pred_inverse
plt.plot(y_testInverse[:100], color='blue', label='Predicted Stock Price')  # Plot first 100 data points of y_test_inverse
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()

"""**1. Root Mean Squared Error (RMSE):** RMSE is a measure of the differences between values predicted by a model and the observed values. It is calculated by taking the square root of the average of the squared differences between the predicted and observed values. RMSE provides a measure of the spread of the residuals (the differences between predicted and observed values), with lower values indicating better fit.

**2. Mean Absolute Percentage Error (MAPE):** MAPE is a measure of the accuracy of a forecasting method in statistics. It is calculated as the average of the absolute percentage differences between the predicted and observed values, divided by the observed values. MAPE provides a measure of the accuracy of the predictions as a percentage of the true values, with lower values indicating better accuracy.
"""

# Calculate RMSE
rmse = np.sqrt(np.mean(((y_pred - y_test) ** 2)))
print('Root Mean Squared Error:', rmse)

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test))

print('Mean Absolute Percentage Error (MAPE):', mape)

"""Now to find the best parameters for the model, we perform grid search cross validation.

"""

def buildModel(optimizer='Adam'):
    model_LSTM = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model_LSTM.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], 1),activation='tanh'))
    model_LSTM.add(Dropout(0.2))
    model_LSTM.add(LSTM(50, return_sequences = True, activation='tanh'))
    model_LSTM.add(Dropout(0.2))
    model_LSTM.add(LSTM(50, return_sequences = True, activation='tanh'))
    model_LSTM.add(Dropout(0.2))
    model_LSTM.add(LSTM(50, activation='tanh'))
    model_LSTM.add(Dropout(0.2))

    model_LSTM.add(Dense(25))
    model_LSTM.add(Dropout(0.2))
    model_LSTM.add(Dense(1))

    model_LSTM.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])

    return model_LSTM

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Create the KerasRegressor with the early stopping callback
model_LSTM = KerasRegressor(model=buildModel,callbacks=[early_stopping])

# Define the hyperparameter grid
param_grid = {
    'optimizer': ['SGD', 'Adagrad', 'Adam'],
    'batch_size': [20, 60, 100,150],
    'epochs': [10, 50, 100]
}
grid = GridSearchCV(estimator=model_LSTM, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train, y_train)


exe_time_nn = time.time() - start_time
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Time taken for execution :", exe_time_nn)

"""We fit the model using the best paramters derived from performing the gris search cross validation."""

st = time.time()
best = model_LSTM.fit(X_train, y_train, epochs = 100, batch_size = 60)
et=time.time()
print("Time to fit: ",(et-st),"s")

"""We plot the $MSE$ over the epochs next for the best paramater model."""

plt.plot(best.history['mean_squared_error'])
plt.title('Loss Function Over Epochs')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.show()

"""There seems to be a significant drop in the maximum value of the $MSE$, being significantly less than the $MSE$ that was seen previously. However, it seems to be more or less stagnant over the number of epochs, indicating that there is not much change that is achieved in the predictions with change in epochs."""

y_pred_best = model_LSTM.predict(X_test)
y_predbest_copy = np.repeat(y_pred_best,7, axis=-1)
y_testbest_copy = np.repeat(y_test,7, axis=-1)

y_predbestInverse = scaler.inverse_transform(np.reshape(y_predbest_copy,(len(y_test),7)))[:,0]
y_testbestInverse = scaler.inverse_transform(np.reshape(y_testbest_copy,(len(y_test),7)))[:,0]

# Create a dataframe for predicted and original values
pred_dfbest = pd.DataFrame({'Predicted Values': y_predbestInverse.flatten(), 'Original Values': y_testbestInverse.flatten()})
print(pred_dfbest)

"""The values seem to be a lot closer than the one from the previous model, thus indicating significant improvement in predictions after performing grid search cross validation."""

# Plot the data
plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.plot(y_predbestInverse[:100], color='red', label='Real Stock Price')  # Plot first 100 data points of y_pred_inverse
plt.plot(y_testbestInverse[:100], color='blue', label='Predicted Stock Price')  # Plot first 100 data points of y_test_inverse
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()

"""The predicted and actual values seem to be a lot closer as compared to before."""

# Calculate RMSE
rmse = np.sqrt(np.mean(((y_pred_best - y_test) ** 2)))
print('Root Mean Squared Error:', rmse)

# Calculate MSE
mse = np.mean(((y_pred_best - y_test) ** 2))
print('Mean Squared Error:', mse)

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred_best) / y_test))
print('Mean Absolute Percentage Error (MAPE):', mape)

#Calculate R^2 score
r2 = r2_score(y_test, y_pred_best)
print('Coefficient of determination (R^2):', r2)

"""There is a drop noticed in both the `rmse` and the `mape` values after implementing the best parameters for training the model.
"""