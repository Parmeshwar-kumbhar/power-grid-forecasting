import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data for the last 30 days (temperature and electricity demand)
np.random.seed(42)
days = np.arange(1, 31)
temperature = np.random.randint(25, 45, size=30)
demand = temperature * 15 + np.random.randint(-50, 50, size=30)

data = pd.DataFrame({
    "Day": days,
    "Temperature": temperature,
    "Demand_Units": demand
})

print("Previous 30 Days Data:")
print(data)

# Visualize the relationship between temperature and demand
plt.style.use("dark_background")
plt.figure(figsize=(10, 5))
plt.scatter(data["Temperature"], data["Demand_Units"], color='blue')
plt.xlabel("Temperature (C)")
plt.ylabel("Electricity Demand (Units)")
plt.title("Temperature vs Electricity Demand")
plt.show()

# Prepare data for the machine learning model
X = data[["Temperature"]]  # Input feature
y = data["Demand_Units"]   # Target variable

# Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict electricity demand for the next 30 days
future_days = np.arange(31, 61)
future_temperature = np.random.randint(25, 45, size=30)

future_data = pd.DataFrame({
    "Day": future_days,
    "Temperature": future_temperature
})

predicted_demand = model.predict(future_data[["Temperature"]])
future_data["Predicted_Demand"] = predicted_demand.round(0)

print("\nNext Month Prediction:")
print(future_data)