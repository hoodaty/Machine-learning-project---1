"""

## COMPARISON BETWEEN MODELS
"""

models_performance = {
    "Linear Regression": {
        "MSE": 3.18e-5,
        "R2": 0.99929,
        "Time": 0.00499
    },
    "Ridge Regression": {
        "MSE": 1.044e-5,
        "R2": 0.9997,
        "Time": 0.27
    },
    "Lasso": {
        "MSE": 1.05e-5,
        "R2": 0.9997,
        "Time": 1.18
    },
    "Decision Tree": {
        "MSE": 2.2e-5,
        "R2": 0.9994,
        "Time": 29.13
    },
    "Random Forest": {
        "MSE": 1.64e-5,
        "R2": 0.9995,
        "Time": 204.37
    },
    "Adaboost": {
        "MSE": 0.00523,
        "R2": 0.9941,
        "Time": 52.74
    },
    "Xgboost": {
        "MSE": 3.15e-5,
        "R2": 0.9941,
        "Time": 8.19
    },
    "LSTM": {
        "MSE": 0.00010,
        "R2": 0.9972,
        "Time": 4357.9

    }
}

"""We display in the form of a dataframe."""

df_comparison = pd.DataFrame(models_performance).T
print(df_comparison)

execution_times = []
mse_values = []
r2_values = []
accuracy_values = []

for model, metrics in models_performance.items():
    execution_times.append(metrics["Time"])
    mse_values.append(metrics["MSE"])
    r2_values.append(metrics["R2"])
    accuracy_values.append(metrics["R2"])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

df_comparison['MSE'].sort_values().plot(kind='bar', ax=axes[0])
axes[0].set_title('Mean Squared Error (MSE)')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('MSE')

df_comparison['R2'].sort_values().plot(kind='bar', ax=axes[1])
axes[1].set_title('R-squared (R2)')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('R2')



plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

df_comparison['Time'].sort_values().plot(kind='bar', ax=axes)
axes.set_title('Time taken (seconds)')
axes.set_xlabel('Model')
axes.set_ylabel('Time')

"""**Based on the comparison of different models:**

**1. MSE (Mean Squared Error):** Lower values indicate better performance in terms of minimizing prediction errors. Among the models, Ridge Regression, Lasso, and Random Forest have relatively lower MSE values, suggesting better predictive accuracy.

**2. R2 (Coefficient of Determination):** Higher values indicate better goodness of fit. Ridge Regression, Lasso, and Random Forest exhibit higher R2 values, indicating a better fit to the data compared to other models.

**3.Time Taken: **LSTM, while providing reasonable predictive performance, requires significantly more computational time compared to other models. Linear Regression and Ridge Regression are the fastest in terms of computation time.


In conclusion, considering both predictive accuracy and computational efficiency, Ridge Regression and Lasso stand out as the top-performing models, followed closely by Random Forest. However, if computational time is not a constraint and higher complexity models are taken into consideration, LSTM may provide competitive performance.

"""