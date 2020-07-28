from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error

# Loading the dataset
dataset = load_boston()

# Describe the dataset
print(dataset.DESCR)

# Load into data frame
dataFrame = pd.DataFrame(dataset.data)

# Fetch column names
dataFrame.columns = dataset.feature_names
dataFrame['ACTUAL_PRICE'] = dataset.target
print (dataFrame.head())

# Statistics
print (dataFrame.describe())

# Correlations
correlations = dataFrame.corr(method='pearson')
print (correlations)

# Visualize correlations
sns.heatmap(data=correlations, cmap="YlGnBu")

# Split features and target
X = dataFrame.drop('ACTUAL_PRICE', axis=1)
Y = dataFrame['ACTUAL_PRICE']

# Train-test split validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

# Train the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Test the model
Y_pred = model.predict(X_test)

# Print RMSE
print(np.sqrt(mean_squared_error(Y_test, Y_pred)))

# Add predicted price to the data frame
dataFrame['PREDICTED_PRICE'] = model.predict(X)

# Plot real price vs predicted price
plotDf = dataFrame.head(50)[["ACTUAL_PRICE", "PREDICTED_PRICE"]]
plotDf.plot(kind='bar',figsize=(16,10))
plt.show()

# Explain results
print(list(zip(X.columns, model.coef_)))
