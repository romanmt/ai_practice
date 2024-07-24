import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load dataset

from sklearn.datasets import fetch_california_housing
california = fetch_california_housing(as_frame=True)
df = california.frame

# data preparation
X = df.drop('MedHouseVal', axis=1)    # Features
y = df['MedHouseVal']                 # Target variagble

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Example new data
new_data = pd.DataFrame({
    'MedInc': [8.3252, 8.3014],  # Median income in block group
    'HouseAge': [41.0, 21.0],    # Median house age in block group
    'AveRooms': [6.9841, 6.2381],# Average number of rooms
    'AveBedrms': [1.0238, 0.9719],# Average number of bedrooms
    'Population': [322.0, 2401.0],# Population in block group
    'AveOccup': [2.5556, 2.1098], # Average occupancy
    'Latitude': [37.88, 37.86],   # Latitude
    'Longitude': [-122.23, -122.22]# Longitude
})

# predict the median hosue values
predictions = model.predict(new_data)

# Print the predictions
print(predictions)
