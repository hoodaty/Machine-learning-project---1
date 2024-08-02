"""
##RIDGE AND LASSO REGRESSION

###Lasso Regression
"""

al = 0.0001# We consider some initial regularization

#We will be training based on all the features irrespective of high correlation, as lasso would deal with it by either minimizing or completely dropping the coefficient
X1 = df_train[['High','Low','Open','Volume','month','year']]
y1 = df_train['Adj Close']
X2 = df_test[['High','Low','Open','Volume','month','year']]
y2 = df_test['Adj Close']

st = time.time()
lasso = Lasso(alpha=al,max_iter=10000)
lasso.fit(X1, y1)
et=time.time()
print("Time to fit: ",(et-st),"s")

# Report coefficients
print('Intercept:', lasso.intercept_)
print('Rounded coef:', list(map(lambda c: round(c, 3), lasso.coef_)))
print('\nR2  (train) = ',lasso.score(X1,y1))
y_pred_train = lasso.predict(X2)

MSE = mean_squared_error(y2, y_pred_train)
print('MSE (train) = ',MSE)

"""We can observe that Lasso has a realtively lower coefficient as compared to simple linear regression. However this is the case where we take an arbitrary regularization. We will perform some cross validation to help us get better results."""

# plotting the results
plt.title('Correlation between actual and predicted values')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.scatter(y2,y_pred_train,color='royalblue')
plt.plot([0,1],[0,1],color='red')
plt.show()

"""We can compute the average $R^{2}$ score and $MSE$ over the arbitrarily chosen $\alpha$ value"""

al = 0.0001
nb_real = 100
#define lists to store R^2 and MSE
R2_test = np.zeros(nb_real)
MSE_test = np.zeros(nb_real)

lasso = Lasso(alpha = al,max_iter=10000)
for real in range(nb_real):#run a for loop in given range to get as many test-train splits
  df_train1, df_test1 = train_test_split(df, train_size = fraction_train, test_size = fraction_test)
  df_train1[:] = scaler.fit_transform(df_train1[:])
  X_train = df_train1[['Open','High','Low','Volume','month','year']]
  y_train = df_train1['Adj Close']
  lasso.fit(X_train, y_train)
  df_test1[:] = scaler.transform(df_test1[:])
  X_test = df_test1[['Open','High','Low','Volume','month','year']]
  y_test = df_test1['Adj Close']
  R2_test[real] = lasso.score(X_test,y_test)
  MSE_test[real] = mean_squared_error(y_test, lasso.predict(X_test))

print('Average R2  (test) =', R2_test.mean())
print('Average MSE (test) =', MSE_test.mean())

"""We make note again that the scores are relatively better as compared to simple linear regression.

Next, we perform some cross validation on the $\alpha$ values to check for the best regularizator value. We also proceed to note the coefficients related to each feature, indicating the feature importance according to LASSO regression.
"""

df_train2, df_test2 = train_test_split(df, train_size = fraction_train, test_size = fraction_test)
df_train2[:] = scaler.fit_transform(df_train2[:])
X_train = df_train2[['Open','High','Low','Volume','year','month']]
y_train = df_train2['Adj Close']

alphas_lasso = np.logspace(-7, -3, 500)
nb_folds = 10
reg = LassoCV(cv=nb_folds, alphas = alphas_lasso,max_iter = 10000)
st = time.time()
reg.fit(X_train, y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
print('Regularizator score (R2 coefficient):', reg.score(X_train, y_train))
y_train_pred = reg.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print('Mean Squared Error (MSE):', mse_train)
print('Regularizator\'s optimal alpha value:', reg.alpha_)
print('Regularizator coefficients:')
[(pair[0], round(pair[1], 4)) for pair in zip(X_train, reg.coef_)]

"""**Observations:**

**1)**The Lasso regression model demonstrates a high level of predictive accuracy, as indicated by the R2 coefficient of 0.9997.

**2)**The optimal alpha value, chosen to balance between model complexity and performance, is 1e-07.

**3)**Notably, the coefficients for 'Open', 'High', and 'Low' exhibit strong correlations with the target variable 'Adj Close', as evidenced by their magnitudes and signs.

**4)**'Volume' does not contribute significantly to the model, as its coefficient is effectively reduced to zero due to Lasso's regularization effect.
The coefficients for 'year' and 'month' are minimal, suggesting limited influence on the target variable compared to the price-related features.\

**5)** $MSE$ is low compared to simple linear regression.

Overall, despite the high collinearity among 'Open', 'High', and 'Low', Lasso regression effectively selects and prioritizes these features for prediction, while disregarding the less impactful 'Volume', 'year', and 'month' variables.

We plot the graph of the mean squared error values versus the $\alpha$ values
"""

mean_mse_arr = np.array(reg.mse_path_).mean(axis=1)
plt.figure(figsize=(10, 8))
plt.semilogx(reg.alphas_, mean_mse_arr, label=feature )
plt.xlabel("Alpha")
plt.ylabel("Mean MSE by fold")
plt.title("Lasso regularizator MSE")
plt.show()

"""There seems to be an increase in the MSE with increase in alpha values, indicating that there is very little regularization that is required."""

# we use coefficients logarithmically spaced in order to cover a wide range of values
alphas_lasso = np.logspace(-5, 1, 100)
coefs_lasso = np.zeros((alphas_lasso.shape[0], X_train.shape[1]))

for i, al in enumerate(alphas_lasso):
    lasso = Lasso(alpha = al,max_iter=10000)
    lasso.fit(X_train, y_train)
    for j, coef in enumerate(lasso.coef_):
        coefs_lasso[i][j] = coef

plt.figure(figsize=(10, 8))
for coef, feature in zip(coefs_lasso.T, X_train.columns):
    plt.semilogx(alphas_lasso, coef, label=feature)
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 0.95))
plt.xlabel("Alpha")
plt.ylabel("Feature weight")
plt.title("Lasso regression")
plt.show()

"""One can notice that the Open feature seems to vanish. This is the case since LASSO decided with the chang over alphas to do so because of the high correlation between the features High, Low and Open. Volume, month and year also have very little impact. We can print the coefficients to verify our claim."""

coefs_lasso

# dataframes need to be converted to numpy format
XX = X_train.to_numpy()
yy = y_train.to_numpy()
_, _, coefs = linear_model.lars_path(XX, yy, method="lasso")

# plot the corresponding result
tau = np.sum(np.abs(coefs.T), axis=1)
plt.figure(figsize=(10, 8))
plt.xlabel("L1 norm of coefficients")
plt.plot(tau, coefs.T, marker="o")
plt.legend(list(X_train.columns), bbox_to_anchor=(1.4, 0.95))
plt.show()

"""In the LARS (Least Angle Regression) path plot, all coefficients, including those that are eventually set to zero, are visualized as the algorithm progresses through the regularization path. This allows you to see how each coefficient evolves with increasing regularization strength. Therefore, even though the "Open" coefficient may eventually be set to zero by Lasso regularization, it still appears in the LARS path plot until it reaches zero.

We can notice that the first feature to be non-zero here is the year possibly since towards the end, we can see a huge spike in adjusted close prices after the year 2020. However, 'High' seems to be the feature with the most relevance as it is the first to be non-zero when decreasing the penalty(starting from a large value).

###Ridge Regression
"""

al = 0.001

X1 = df_train[['High','Open','Low','Volume','year','month']]
y1 = df_train['Adj Close']
X2 = df_test[['High','Open','Low','Volume','year','month']]
y2 = df_test['Adj Close']

st = time.time()
ridge = Ridge(alpha=al,max_iter=10000)
ridge.fit(X1, y1)
et=time.time()
print("Time to fit: ",(et-st),"s")

# Report coefficients
print('Intercept:', ridge.intercept_)
print('Rounded coef:', list(map(lambda c: round(c, 3), ridge.coef_)))
print('\nR2  (train) = ',ridge.score(X1,y1))
y_pred_train = ridge.predict(X2)

MSE = mean_squared_error(y2, y_pred_train)
print('MSE (train) = ',MSE)

"""Here using an arbitrary regularizator value gives us a better result in terms of score and $MSE$."""

# plotting the results
plt.title('Correlation between actual and predicted values')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.scatter(y2,y_pred_train,color='royalblue')
plt.plot([0,1],[0,1],color='red')
plt.show()

"""We calculate the average $MSE$ and $R^{2}$ score over various test train splits"""

al = 0.01
nb_real = 100

R2_test = np.zeros(nb_real)
MSE_test = np.zeros(nb_real)

ridge = Ridge(alpha = al,max_iter=10000)
for real in range(nb_real):
  df_train1, df_test1 = train_test_split(df, train_size = fraction_train, test_size = fraction_test)
  df_train1[:] = scaler.fit_transform(df_train1[:])
  X_train = df_train1[['Open','High','Low','Volume','year','month']]
  y_train = df_train1['Adj Close']
  ridge.fit(X_train, y_train)
  df_test1[:] = scaler.transform(df_test1[:])
  X_test = df_test1[['Open','High','Low','Volume','year','month']]
  y_test = df_test1['Adj Close']
  R2_test[real] = ridge.score(X_test,y_test)
  MSE_test[real] = mean_squared_error(y_test, ridge.predict(X_test))

print('Average R2  (test) =', R2_test.mean())
print('Average MSE (test) =', MSE_test.mean())

"""Next, we perform some cross validation on the  α  values to check for the best regularizator value. We also proceed to note the coefficients related to each feature, indicating the feature importance according to Ridge regression."""

df_train2, df_test2 = train_test_split(df, train_size = fraction_train, test_size = fraction_test)
df_train2[:] = scaler.fit_transform(df_train2[:])
X_train = df_train2[['Open','High','Low','Volume','year','month']]
y_train = df_train2['Adj Close']

alphas_ridge = np.logspace(-7, -3, 500)
reg = RidgeCV(cv=None, alphas = alphas_ridge, scoring=None, store_cv_values=True)
st = time.time()
reg.fit(X_train, y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
print('Regularizator score (R2 coefficient):', reg.score(X_train, y_train))
y_train_pred = reg.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print('Mean Squared Error (MSE):', mse_train)
print('Regularizator\'s optimal alpha value:', reg.alpha_)
print('Regularizator coefficients:')
[(pair[0], round(pair[1], 4)) for pair in zip(X_train.columns, reg.coef_)]

"""**Observations:**

**1)**The Ridge regression model demonstrates a high level of predictive accuracy, as indicated by the R2 coefficient of 0.9997.

**2)**The optimal alpha value, chosen to balance between model complexity and performance, is 1e-07. Thus indicating that very little regularization is required.

**3)**Notably, the coefficients for 'Open', 'High', and 'Low' exhibit strong correlations with the target variable 'Adj Close', as evidenced by their magnitudes and signs.

**4)**'Volume' does not contribute significantly to the model, as its coefficient is effectively reduced to zero due to Lasso's regularization effect.
The coefficients for 'year' is 0 and 'month' is minimal, suggesting limited influence on the target variable compared to the price-related features.

We plot the graph of the mean squared error values versus the  $\alpha$  values
"""

mean_mse_arr = np.array(reg.cv_values_).mean(axis=0)
plt.figure(figsize=(10, 8))
plt.semilogx(alphas_ridge, mean_mse_arr, label=feature )
plt.xlabel("Alpha")
plt.ylabel("Mean MSE by fold")
plt.title("Ridge regularizator MSE")
plt.show()

"""There seems to be an increase in the MSE with increase in alpha values, indicating that there is very little regularization that is required."""

alphas_ridge = np.logspace(-5, 4, 100)
coefs_ridge = np.zeros((alphas_ridge.shape[0], X_train.shape[1]))

for i, al in enumerate(alphas_ridge):
    ridge = Ridge(alpha = al)
    ridge.fit(X_train, y_train)
    for j, coef in enumerate(ridge.coef_):
        coefs_ridge[i][j] = coef

plt.figure(figsize=(10, 8))
for coef, feature in zip(coefs_ridge.T, X_train.columns):
    plt.semilogx(alphas_ridge, coef, label=feature)
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 0.95))
plt.xlabel("Alpha")
plt.ylabel("Feature weight")
plt.title("Ridge regression")
plt.show()

"""From this plot, it is evident that 'High' has the largest coefficient indicating having the largest influence in predicting the adjusted close prices, followed by 'Low' and 'Open'.
"""