import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

raw_data = {
    'age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49, 55, 25, 58, 19, 18, 21, 26, 40, 45, 50, 54, 23],
    'bought_insurance': [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
}
df = pd.DataFrame(raw_data)

X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.9, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print("=== Scikit-Learn Logistic Regression ===")
print(f"Test Set Ages:\n{X_test}")
print(f"Predicted Outcomes (0 = No, 1 = Yes): {predictions}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(8, 5))
plt.scatter(df.age, df.bought_insurance, marker='+', color='red', label='Actual Data (0 or 1)')

age_range = np.linspace(15, 65, 100).reshape(-1, 1)
probs = model.predict_proba(age_range)[:, 1]
plt.plot(age_range, probs, color='blue', linewidth=2, label='Fitted Logistic (Sigmoid) Curve')

plt.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
plt.xlabel('Age')
plt.ylabel('Probability of Buying Insurance')
plt.title('Logistic Regression using Scikit-Learn (Codebasics)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
