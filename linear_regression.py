import numpy as np
import matplotlib.pyplot as plt

# Reproducible random data
np.random.seed(42)

# 200 data points
X = np.random.uniform(1, 10, 200)

# Noise
noise = np.random.normal(0, 5, 200)

# Generate y
y = 8 * X + 20 + noise

# OLS Linear Regression

# Mean
X_mean = np.mean(X)
y_mean = np.mean(y)

# Slope (w)
w = np.sum((X - X_mean) * (y - y_mean)) / \
    np.sum((X - X_mean) ** 2)

# Intercept (b)
b = y_mean - w * X_mean


# Predictions
y_pred = w * X + b


# MSE
mse = np.mean((y - y_pred) ** 2)


# Results
print("Weight (w):", w)
print("Bias (b):", b)
print("MSE:", mse)


# Prediction for new value
hours = 7

prediction = w * hours + b

print("Prediction for", hours, "hours:", prediction)

# Graph

plt.scatter(X, y, label="Actual Data")

# Sort X so line looks proper
sort_index = np.argsort(X)

plt.plot(
    X[sort_index],
    y_pred[sort_index],
    label="OLS Regression Line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("OLS Linear Regression")

plt.legend()
plt.show()
