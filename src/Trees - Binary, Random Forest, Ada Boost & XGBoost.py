"""
##TREES

###Decision Trees

We start by performing regression with a single tree.

The key parameters to avoid overfitting are
- *max_depth* to limit the maximal depth of the tree
- *min_samples_leaf* to ask for a minimal number of data points for a condition to be satisfied
We start by playing around with a few values of these parameters to get a feeling of what is done/obtained.
"""

# choice of parameters
chosen_max_depth = 5
chosen_min_samples_leaf = 4

# constructing + fitting the model and making the prediction
st = time.time()
dt = DecisionTreeRegressor(max_depth=chosen_max_depth, min_samples_leaf=chosen_min_samples_leaf)
model = dt.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
y_pred = model.predict(X_test)

# computing the performance on the test set
# MSE between prediction and true values
mse = mean_squared_error(y_pred,y_test)
print('MSE (test):',mse)
# coefficient of determination for the regression
R2 = dt.score(X_test,y_test)
print('R2  (test):',R2)
# plotting predicted values as a function of true values to graphically assess the quality of regression
plt.title('Correlation between actual and predicted values')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.scatter(y_test,y_pred,color='royalblue')
plt.plot([0,1],[0,1],color='red')
plt.show()

"""The plot depicting the correlation between actual and predicted values shows a strong linear relationship, with predicted values closely aligning with the diagonal line representing perfect prediction. Overall, the performance of the Decision Tree Regression model appears to be comparable to that of a simple linear regression model.

We can visualize the tree using the function plot_tree

The performance seems to be comparable to that of simple linear regression
"""

print('Number of data points : ',y_train.shape[0])
for pair in zip(df_train.columns, np.arange(df_train.shape[0])):
  print('X[',pair[1],'] = ', pair[0])
plt.figure(figsize=(15, 10))
tree.plot_tree(dt)
plt.show()

"""We next explore more systematically the choice of the parameters by cross validation using GridSearchCV"""

dt = DecisionTreeRegressor()
dt_params = {'max_depth':np.arange(1,20),'min_samples_leaf':np.arange(1,20)}

# the output (best parameters) is in the 'dict' format = dictionary
print('Grid search to find optimal parameters')
st = time.time()
gs_dt = GridSearchCV(estimator=dt, param_grid=dt_params)
gs_dt.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
a = gs_dt.best_params_
print('- Best maximal depth =',a['max_depth'])
print('- Best minimal number of samples in the leaves = ',a['min_samples_leaf'],'\n')

# training with best parameters
st = time.time()
dt_final = DecisionTreeRegressor(max_depth=a['max_depth'], min_samples_leaf=a['min_samples_leaf'])
model = dt_final.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
y_pred = model.predict(X_test)

# computing the performance on the test set
# MSE between prediction and true values
mse = mean_squared_error(y_pred,y_test)
print('MSE (test):',mse)
# coefficient of determination for the regression
R2 = dt_final.score(X_test,y_test)
print('R2  (test):',R2)
# plotting predicted values as a function of true values to graphically assess the quality of regression
plt.title('Correlation between actual and predicted values')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.scatter(y_test,y_pred,color='royalblue')
plt.plot([0,1],[0,1],color='red')
plt.show()

"""We can see a decrease in the $MSE$ and a higher $R^2$ score. Prediction wise it seems to be comparable to Lasso regression. However, we use trees for better interpretability of the data. If the data were to be non-linear, using trees would have displayed better results.

###Random Forests

We will use random forests by combining trees and see if there is any difference in results.
"""

rf = RandomForestRegressor()

rf_params = {'n_estimators':np.arange(25,150,25),'max_depth':np.arange(1,11,2),'min_samples_leaf':np.arange(2,15,3)}
print('Grid search to find optimal parameters')
st = time.time()
gs_rf = GridSearchCV(estimator=rf,param_grid=rf_params)
gs_rf.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
b = gs_rf.best_params_
print('- Best number of trees = ',b['n_estimators'])
print('- Best maximal depth =',b['max_depth'])
print('- Best minimal number of samples in the leaves = ',b['min_samples_leaf'],'\n')

# fitting the model with best params
RF = RandomForestRegressor(n_estimators=b['n_estimators'],max_depth=b['max_depth'],min_samples_leaf=b['min_samples_leaf'])
st = time.time()
model = RF.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
y_pred = model.predict(X_test)

# compute the performance on the test set
R2 = RF.score(X_test,y_test)
print('R2  (test):',R2)
mse = mean_squared_error(y_pred,y_test)
print('MSE (test):',mse)
plt.title('Correlation between actual and predicted values')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.scatter(y_test,y_pred,color='royalblue')
plt.plot([0,1],[0,1],color='red')
plt.show()

"""We can see an improvement in performance with scores being comparable to Ridge regression.

###AdaBoost

Next we try and see if there is any change in performance with AdaBoost.
"""

ar = AdaBoostRegressor(estimator = dt_final)

print('Tree maximal depth (weak learner) =',dt_final.max_depth)
print('Minimal number of samples in the leaves (weak learner) = ',dt_final.min_samples_leaf,'\n')
# key parameter: number of weak learners to be considered
print('Grid search to find optimal parameters')
ar_params = {'n_estimators':np.arange(10,200,10)}
st = time.time()
gs_ar = GridSearchCV(estimator=ar,param_grid=ar_params)
gs_ar.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
c = gs_ar.best_params_
print('- Best number of weak learners (trees) = ',c['n_estimators'],'\n')

# Fitting the model with best params
st = time.time()
ab_dt = AdaBoostRegressor(n_estimators=50)
model = ab_dt.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
y_pred = model.predict(X_test)

# computing the performance on the test set
R2 = ab_dt.score(X_test,y_test)
print('R2  (test):',R2)
mse = mean_squared_error(y_pred,y_test)
print('MSE (test):',mse)
plt.title('Correlation between actual and predicted values')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.scatter(y_test,y_pred,color='royalblue')
plt.plot([0,1],[0,1],color='red')
plt.show()

"""The performance is significantly weaker in this case, as compared to all other models that were used before. This is because of multicollinearity. AdaBoost's performance can be influenced by correlated features, especially if they dominate the feature importance in the ensemble.

###XGBoost
"""

xgboost = XGBRegressor()

print('Tree maximal depth (weak learner) =', dt_final.max_depth)
print('Minimal number of samples in the leaves (weak learner) = ', dt_final.min_samples_leaf, '\n')

print('Grid search to find optimal parameters')
xgboost_params = {'n_estimators': range(10, 200, 10)}
st = time.time()
gs_xgboost = GridSearchCV(estimator=xgboost, param_grid=xgboost_params)
gs_xgboost.fit(X_train, y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
d = gs_xgboost.best_params_
print('- Best number of weak learners (trees) = ', d['n_estimators'], '\n')

# Fitting the model with best params
st = time.time()
xg_dt = XGBRegressor(n_estimators=30)
model = xg_dt.fit(X_train,y_train)
et=time.time()
print("Time to fit: ",(et-st),"s")
y_pred = model.predict(X_test)

# computing the performance on the test set
R2 = ab_dt.score(X_test,y_test)
print('R2  (test):',R2)
mse = mean_squared_error(y_pred,y_test)
print('MSE (test):',mse)
plt.title('Correlation between actual and predicted values')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.scatter(y_test,y_pred,color='royalblue')
plt.plot([0,1],[0,1],color='red')
plt.show()

"""Although the $MSE$ is quite low compared to the AdaBoost, the $R^{2}$ is comparable."""

plt.figure(figsize=(10, 8))
plot_importance(xg_dt, max_num_features=6)
plt.title("Feature Importance")
plt.show()

"""From this it seems that XGBoost migh be using the volume feature to split the data at the root node. This is probably because that feature is the only one where the values are not correlated, other than month or year.
"""