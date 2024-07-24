import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data with a categorical feature
data = {
    'Neighborhood': ['A', 'B', 'A', 'B', 'C'],
    'Size': [1500, 1600, 1700, 1800, 1900],
    'Price': [300000, 320000, 330000, 350000, 360000]
}

df = pd.DataFrame(data)

# Convert categorical variable to dummy/indicator variables
df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

# Prepare features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Train-test split and model training as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(df)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
