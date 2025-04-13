import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib  

# Set time range
start = datetime(2005, 1, 1)
end = datetime(2025, 1, 1)

# FRED series codes for various indicators
symbols = {
    "HomePrice": "CSUSHPISA",        # S&P Case-Shiller
    "MortgageRate": "MORTGAGE30US",  # 30-Year Mortgage Rate
    "UnemploymentRate": "UNRATE",    # Unemployment Rate
    "CPI": "CPIAUCSL",               # Inflation
    "FedFundsRate": "FEDFUNDS",      # Federal Funds Rate
    "BuildingPermits": "PERMIT",     # Building Permits
    "ConsumerSentiment": "UMCSENT"   # Consumer Sentiment
}

# Download data
data = pd.DataFrame()
for name, code in symbols.items():
    data[name] = web.DataReader(code, 'fred', start, end)

# Monthly average, clean missing values
data = data.resample('M').mean().dropna()

# ---------------------
# Multiple Linear Regression

# Features and target
X = data.drop(columns=['HomePrice'])
y = data['HomePrice']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')  

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# Coefficients interpretation
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Impacts on Home Prices:")
print(coefficients)

# Plot predicted vs actual
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Home Prices")
plt.ylabel("Predicted Home Prices")
plt.title("Actual vs Predicted Home Prices")
plt.tight_layout()
plt.show()

# Correlation heatmap (optional)
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
