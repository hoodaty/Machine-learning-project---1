"""EXPLORATORY DATA ANALYSIS

Next, we perform some Exploratory Data Analysis on the dataset where we plot graphs and check for redundant features or multicollinearity.
"""

df = pd.read_csv('TSLA.csv')#reads the datafile
df.info()#providessome basic information about the dataset

print(df.shape)#gives us the shape of the dataset, number of samples x number of features

df.head()#prints the initial few values of the dataset

df.describe()#provides statsitics regarding the features in the datatset based on the samples

"""**Observations:**

**1) Open, High, Low, Close, and Adj Close Prices:**
The mean values for these prices indicate that the stock tends to have a high average price, with the Close and Adj Close prices being very close.
The standard deviation values suggest that there is a considerable amount of variability or volatility in these prices, as the standard deviations are quite large compared to the mean values.

**2) Volume:**
The mean volume indicates the average number of shares traded, which is relatively high, suggesting that there is significant trading activity for this stock.
The standard deviation of volume suggests that the trading volume can vary considerably from day to day.

**3) Minimum and Maximum Values:**
The minimum and maximum values show the range within which the prices and volume have fluctuated over the given period.
The minimum values provide insights into the lowest prices and volumes observed, while the maximum values indicate the highest prices and volumes observed.

**4) Percentiles (25th, 50th, and 75th):**
These percentiles provide information about the distribution of prices and volume.
For example, the 25th percentile (also known as the first quartile) represents the value below which 25% of the data falls.
Similarly, the 50th percentile (median) represents the value below which 50% of the data falls, and the 75th percentile (third quartile) represents the value below which 75% of the data falls.

We index the dataset using the 'Date' column. The information that would be required from this column for regression purposed are the months and the year of the datapoint, which will be extracted and encoded.
"""

df['Date'] = pd.to_datetime(df['Date'])#Convert the 'Date' column into an index
# Set 'Date' as the index
df.set_index('Date', inplace=True)

df['year'] = df.index.year#extract the year and add an additional feature to the dataset
df['month'] = df.index.month#extract the month and add an additional feature to the dataset

"""We use label encoding to enclode the year feature, since they are ordinal in nature and more recent years would play a comparitively higher role in predicting the values of the target feature which is the adjusted close feature in our case."""

label_encoder = LabelEncoder()
df['year'] = label_encoder.fit_transform(df['year'])#usinglabel encoder to enclode the years for easier interpretability

"""### Data Visualisation

Next, we plot a few graphs to visualise the data helping us to interpret the data better.
"""

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
# Enumerate over all columns
for idx, feature in enumerate(df.columns):
    x = df.index  # Extracting the datetime index as x-values
    y = df[feature]  # Selecting the column values as y-values
    axes[idx // 4, idx % 4].scatter(x, y)  # Plotting scatter plot
    axes[idx // 4, idx % 4].set_title(feature)  # Setting title for each subplot
plt.tight_layout()  # Adjusting layout
plt.show()

"""**Observations:**

**1) Similarity in Open, High, Low, Close, and Adj Close vs Adj Close:**
The plots for Open, High, Low, Close, and Adj Close appear to be very similar, which is expected as these variables are typically highly correlated in stock market data. This similarity suggests that the price movements for these variables are closely related over time.

**2) Relationship between volume and adjusted close prices:**
The plot for the Volume feature shows some variability over time, with certain periods of higher trading volume and others of lower volume. This variability indicates fluctuations in trading activity in the market.
Additionally, there seems to be a positive correlation between Volume and time, as the volume tends to increase over the years. This suggests that trading activity has generally increased over time.

**3) Relationship between year and adjusted close prices:**
The observation of a clear positive correlation with the Year feature confirms the previous point about increasing trading activity over the years. This indicates a potential long-term trend in the market.
The positive correlation with the Year feature suggests that there may be some underlying factors or trends driving the increase in trading activity over time.

**4) Relationship between month and adjusted close prices:**

We cannot derive a proper explaination from this graph regarding the relationship between the month and the adjusted close prices. However, the thickness of the data points do indicate that there has been a rise in the prices but again that is not too clear.
"""

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
# position in the axes: vector [i,j] with i is the line number, and j the column index
for idx, feature in enumerate(df.columns):
    df.plot(feature, "Adj Close", subplots=True,  kind="scatter", ax=axes[idx // 4, idx % 4])
plt.show()

"""**Observations:**

**1) Linear Relationship between Open, High, Low, Close, and Adj Close:**
There appears to be a linear relationship between the Open, High, Low, Close, and Adj Close prices. This linear relationship is expected.

**2) Volume and adjusted close Price Relationship:**
There seems to be an inverse relationship between the volume traded and the price of the shares. As the price of the shares increases, the volume traded tends to decrease, and vice versa. This observation is consistent with typical market behavior, where higher prices may lead to decreased trading activity and vice versa.

**3) Year and adjusted close price relationship:**
The values of the Adj Close price seem to be increasing over the years, with a noticeable spike in the last two years. This suggests a potential trend of increasing minimum prices over time, indicating overall growth or inflation in stock prices.

**4) Month and adjusted close Price Relationship:**
There appears to be an upward trend in the Adj Close prices with changes in months across different years. This suggests that there may be seasonal trends or patterns in stock prices, with certain months exhibiting higher prices compared to others.

Next we plot the trend of the Open, High, Close, Adj Close and Volume with respect to time
"""

plt.figure(figsize=(12, 12))

columns = [str(c) for c in df[['Open','High','Close','Adj Close','Volume']]]#declaring the columns to be used for the subplot

for i, c in enumerate(columns):
    plt.subplot(len(columns), 1, i+1)
    plt.plot(df[c])
    plt.title(c, y=0.5, loc='left')

plt.show()

"""There seems to be very little increase in the values of the prices, other than volume which has a few spikes in between, till the year 2020, after which there is a significant rise in the share prices. This can possibly be related due to trends related to the company during recent years because of which there might have been some exceptional engagement. It can also be attributed to the COVID-19 pandemic.

We plot all the prices to observe how closely they are related
"""

plt.figure(figsize = (18,10))
plt.plot(df['Open'])
plt.plot(df['Adj Close'])
plt.plot(df['Close'])
plt.plot(df['High'])
plt.plot(df['Low'])
plt.legend(['Open','Adj Close','Close','High','Low'])

"""This graph again confirms the high correlation that is present between all of the price features in the dataset which is a trademark for almost all relevant stocks in the share market.

We will drop the 'Close' column, since it is unnecessary to compare it with the adjusted closing prices.
"""

df.drop('Close',axis=1,inplace=True)

"""We create additional columns in plot_df(copy of df) to extract various time-related features from the index of the dataset. These features include:

* **weekday_name:** Name of the weekday (e.g., Monday, Tuesday).

* **weekday:** Numeric representation of the weekday (0 for Monday, 1 for Tuesday, and so on).

* **week:** Week number of the year.

* **day:** Day of the month.

* **hour:** Hour of the day
.
* **date:** Date component of the index.

* **month:** Month number.
* **month_name:** Name of the month (e.g., January, February).

* **year:** Year component of the index.

We convert the month_name and weekday_name columns to ordered categoricals to ensure that they are sorted correctly in plots.
"""

#create a copy of dataset, then define indices
plot_df = df.copy()
plot_df["weekday_name"] = plot_df.index.day_name()
plot_df["weekday"] = plot_df.index.weekday
plot_df["week"] = plot_df.index.isocalendar().week
plot_df["day"] = plot_df.index.day
plot_df["hour"] = plot_df.index.hour
plot_df["date"] = plot_df.index.date
plot_df["month"] = plot_df.index.month
plot_df["month_name"] = plot_df.index.month_name()
plot_df["year"] = plot_df.index.year
#Although many of these are not features of the index columns 'Date', I still declare them for flexibility in future analysis, in case the dataset gets updated.
#Making ordered categoricals to make for sorted plots
plot_df['month_name'] = pd.Categorical(plot_df['month_name'], categories=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], ordered=True)
plot_df['weekday_name'] = pd.Categorical(plot_df['weekday_name'], categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)

"""We group plot_df by year and month_name, calculate the mean of the 'Adj Close' prices for each group, drop any missing values, and reset the index. The resulting DataFrame plot_df_monthly contains the average 'Adj Close' prices for each month across different years."""

plot_df_monthly = plot_df[plot_df.year.isin([2016, 2017,2018, 2019, 2020, 2021, 2022])].groupby(['year', 'month_name'])['Adj Close'].mean().dropna().reset_index()
plot_df_monthly

fig = px.line(plot_df_monthly, x="month_name", y='Adj Close', color="year", title="Yearwise Monthly Plot")
fig.show()

"""Below, I have also plotted the Pearson heatmap to confirm the correlation between the features"""

plt.figure(figsize = (15,10))
sns.heatmap(df.corr(method='pearson'), annot = True, linewidths=.5, cmap="coolwarm")
plt.show()

"""Thus we can now confirm the high correlation between the 'Open', 'High', 'Low' and 'Adj Close' prices. They year also has a large positive correlation with the other features except the months, which is the only feature with negative correlations with all of the other features.

##NORMALISATION OF DATA

Now we split the data into testing and training sets in a 70:30 ratio
"""

fraction_train = 0.7
fraction_test = 1.0 - fraction_train
df_train, df_test = train_test_split(df, train_size = fraction_train, test_size = fraction_test)
df_train.info()
df_test.info()

"""Using MinMaxScalar, we can normalize the training data"""

scaler = MinMaxScaler()
df_train[:] = scaler.fit_transform(df_train[:])
df_train.describe()

"""We print the minimum and maximum values to make sure that the normalization takes place properly"""

min_values = df_train.min()
max_values = df_train.max()

print("Minimum values of features in the test data:")
print(min_values)

print("\nMaximum values of features in the test data:")
print(max_values)

"""Similarly, we perform the same commands for the testing data"""

df_test[:] = scaler.transform(df_test[:])
df_test.describe()

min_values = df_test.min()
max_values = df_test.max()

print("Minimum values of features in the test data:")
print(min_values)

print("\nMaximum values of features in the test data:")
print(max_values)

"""We can see that there are features which are not in the range $[0,1]$. This is because we are transforming the test data based on the fitting which was done on the training data.
"""