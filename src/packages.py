# -*- coding: utf-8 -*-
"""HOODATY_Soumodeep_ML.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BttZ9BGsVyry8Fp3Wp0g_QpDuRpA5jdD

#REGRESSION ANALYSIS ON TESLA STOCK PRICES

![download.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAA4CAMAAACfWMssAAAAdVBMVEX////lGDfjABXjABjjAB7kACzraHXkACbjABL98/TkACjkACPlEjTvi5Pzsbb51tjqYm/40dPmMEbte4XiAAD0uLzoTFzjAAn++Pn63+Hsb3r86uvwkpryqK7wl5764+PmJj7nOUzug4zxoKbpVWT2xsnoRFbELkyZAAAB0ElEQVRIie2W63KDIBBGkZtBQKJGvMXEmMT3f8Qu0nY6nWKDf9szoxjGI2ZnPxKE/iZtMWpuS5WmqrRcj0X7giTqajrhw+GAycp6eZqqWmxao20Ixrc5L4b3VdqhyOcbxqRZxqDbnShRukDI5GfO/VzPdW4QKrQi9GRCpu0NMrpsZMZU2ruZSiqWyaaEx5n+Fn7XQWeUqSRRcNA7QiOB6wQOJtN52PiScwN3MQrVhBEXBsMwqZRmMDbVhojOByXP7tFj5tfKRvciF6mw3vKgClPvR5msyOf6sZqKbQ/wRRfYi1h8mXyJ9UXh9LrxzsOLj2hRMyeyX0ryA3XqxDSPFjvqRBrssiCtF19J1DdurvXKeA/1UB3Fd4h3aLrsskN8ys+Gi2OApiNbUQoCoWR7PGSVWnaJM2Ob2Q2Sp2uK4zGU/p7en2gpicjvV6b4FHuO8Sn2nONT7KnjU+zp9hUVyrojxf/8PWrJJINNxjClMvjjcJFTkkDTJb5/epieAvm8rvu2GByQjVwIAVLpReEIrNkdV91au9Sw5djHo9SfoloWSwK/XV4crg73hOt1dKLpOhDcFtsHxOG+njnnR9j68yNcwMoaBvDdTnnvNqv0zwdvpdwYp13xMyYAAAAASUVORK5CYII=)

## IMPORTING REQUIRED PACKAGES & DATASET

We first import all the relevant packages that would be required for implementing the desired methods for regression.
"""

!pip install scikeras #to install the scikeras package

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
# %matplotlib inline
import time
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from xgboost.plotting import plot_importance
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import matplotlib.dates as mdates
from keras.optimizers import SGD, Adagrad, Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# set up the random number generator: given seed for reproducibility, None otherwise
# (see https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng)
my_seed = 1
rng = np.random.default_rng(seed=my_seed)