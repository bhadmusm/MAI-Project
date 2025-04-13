#===================================LINEAR REGRESSION CPU===============#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# start timing
start_time = time.time()

# load dataset and process
file_path = 'normalized_allDataMeanFINAL_PUE.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# define features and target
features = ['IT Equipment Power (Watts)',
            'RAM memory power consumption - Percentage (%)',
            'GPU consumption - Percentage (%)',
            'CPU temperature - Centigrade Degrees (Â°C)',
            'RAM temperature - Centigrade Degrees (Â°C)',
            'GPU temperature - Centigrade Degrees (Â°C)',
            'RAM memory consumption - Percentage (%)',
            'Power (PA) - Watts (W)',
            'Current (A)']
target_column = 'CPU power consumption - Percentage (%)'

# clean column names and check
df.columns = df.columns.str.strip()
print(df.columns.tolist())

# drop rows with missing values
col_req = features + [target_column]
df = df.dropna(subset=col_req)

# scaling features and target
x_scaled_val = MinMaxScaler()
X_all = x_scaled_val.fit_transform(df[features])

y_scaler = MinMaxScaler()
y_all_vals = y_scaler.fit_transform(df[[target_column]])

# train-test split (no filtering yet)
length_of_training = int(len(X_all) * 0.7)
val_len = int(len(X_all) * 0.15)

X_train = X_all[:length_of_training]
y_train = y_all_vals[:length_of_training]

X_val = X_all[length_of_training:length_of_training + val_len]
y_val = y_all_vals[length_of_training:length_of_training + val_len]

X_test = X_all[length_of_training + val_len:]
y_test = y_all_vals[length_of_training + val_len:]

# fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# inverse transform predictions and targets
y_test_orig = y_scaler.inverse_transform(y_test)
y_pred_orig = y_scaler.inverse_transform(y_pred)

# === filter for active data (after prediction) ===
active_mask = y_test_orig.flatten() > 0.01
y_test_active = y_test_orig[active_mask]
y_pred_active = y_pred_orig[active_mask]

# === evaluation metrics ===
rmse = np.sqrt(mean_squared_error(y_test_active, y_pred_active))
mae = mean_absolute_error(y_test_active, y_pred_active)
r2 = r2_score(y_test_active, y_pred_active)

print(f"\nLinear regression CPU power consumption metrics")
print(f"Values for rmse     : {rmse:.4f}")
print(f"Values for mae      : {mae:.4f}")
print(f"Values forr²       : {r2:.4f}")

# time taken
elapsed = time.time() - start_time
print(f"The total execution time: {elapsed:.2f} seconds")

# === plotting ===
plt.figure(figsize=(12, 6))
plt.plot(y_test_active, label='Actual cpu power consumption', linestyle='--', alpha=0.7, color='dodgerblue')
plt.plot(y_pred_active, label='Predicted cpu power consumption', alpha=0.6, color='hotpink')
plt.title("Linear regression plot: actual vs predicted")
plt.xlabel("Time steps(in seconds)")
plt.ylabel("Cpu power consumption (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("Actual cpu power (normalized)")
plt.ylabel("Predicted cpu power (normalized)")
plt.title("Actual vs. cpu power (scatter plot)")
plt.grid(True)
plt.tight_layout()
plt.show()




