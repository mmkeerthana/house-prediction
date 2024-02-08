import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read the data
your_data = pd.read_csv('data.csv')

# Drop columns that are not needed for the prediction
X = your_data.drop(['date', 'price'], axis=1)
y = your_data['bedrooms']

# Separate numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', 'passthrough', categorical_cols)  # Handle categorical columns separately
    ]
)

# Preprocessing the data
X_preprocessed = preprocessor.fit_transform(X)

# One-hot encoding for categorical columns using pd.get_dummies
X_preprocessed = pd.get_dummies(pd.DataFrame(X_preprocessed, columns=numerical_cols.tolist() + categorical_cols.tolist()))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr}')

# Neural Network
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predict with the neural network
y_pred_nn = nn_model.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network MSE: {mse_nn}')

# Plotting actual vs predicted for Linear Regression
plt.scatter(y_test, y_pred_lr)
plt.xlabel('Actual Bedrooms')
plt.ylabel('Predicted Bedrooms')
plt.title('Linear Regression: Actual vs Predicted Bedrooms')
plt.show()

# Plotting actual vs predicted for Neural Network
plt.scatter(y_test, y_pred_nn)
plt.xlabel('Actual Bedrooms')
plt.ylabel('Predicted Bedrooms')
plt.title('Neural Network: Actual vs Predicted Bedrooms')
plt.show()
