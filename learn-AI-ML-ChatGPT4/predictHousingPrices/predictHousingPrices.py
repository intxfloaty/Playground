import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a simple dataset
data = {
    'size(sq ft)': [1500,2000,2500,3000,3500,4000],
    'price(USD)' : [100000, 140000, 180000, 220000, 260000, 300000]
}

df = pd.DataFrame(data)


# Split the data into training and testing sets
X = df[['size(sq ft)']]
y = df[['price(USD)']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Eroor: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

#Test the model

new_house_size = 1350
predicted_price = model.predict([[new_house_size]])
print(f'Predicted price for a {new_house_size} sqft house: {predicted_price[0].item():.2f}')