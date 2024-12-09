# Polynomial Regression Model
# ======================================================
# This script implements a polynomial regression model using PyTorch.
# It includes data preprocessing, model training, evaluation, and
# statistical analysis. Results are saved as plots and model summaries.

# ======================================================
# 1. Import Libraries
# ======================================================
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import joblib
import sys

sys.path.append("..")
import utility.plot_settings

# Check for GPU availability for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================
# 2. Read Data and Split into Train/Test Sets
# ======================================================
# Load the housing dataset
df = pd.read_pickle("../../data/interim/housing.pkl")
df = pd.DataFrame(df)

# Split the data into training (75%) and testing (25%) sets
df_train, df_test = train_test_split(
    df, train_size=0.75, test_size=0.25, random_state=100
)

# Separate the target variable ('price') from the features
y_train = df_train.pop("price")
X_train = df_train
y_test = df_test.pop("price")
X_test = df_test

# ======================================================
# 3. Normalize Data and Create Polynomial Features
# ======================================================
# Initialize MinMaxScaler for scaling features and target
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Identify numerical features for normalization
numerical_list = [
    x for x in X_train.columns if X_train[x].dtype in ("int64", "float64")
]

# Scale the numerical features to [0, 1] range
X_train[numerical_list] = scaler_features.fit_transform(X_train[numerical_list])
X_test[numerical_list] = scaler_features.transform(X_test[numerical_list])

# Scale the target variable (price)
y_train_scaled = scaler_target.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_target.transform(y_test.values.reshape(-1, 1))

# Apply polynomial feature transformation
degree = 5  # Degree of polynomial features
poly_reg = PolynomialFeatures(degree=degree, include_bias=False)

X_train_poly = pd.DataFrame(poly_reg.fit_transform(X_train), index=X_train.index)
X_test_poly = pd.DataFrame(poly_reg.transform(X_test), index=X_test.index)

# Convert data to PyTorch tensors for training
X_train_scaled = torch.tensor(X_train_poly.values, dtype=torch.float32).to(device)
y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_scaled = torch.tensor(X_test_poly.values, dtype=torch.float32).to(device)
y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)


# ======================================================
# 4. Define the Polynomial Regression Model
# ======================================================
# This is a simple linear regression model designed for polynomial features
class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(PolynomialRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Fully connected layer

    def forward(self, x):
        return self.fc(x)


# Initialize the model, loss function, and optimizer
input_dim = X_train_scaled.shape[1]  # Number of input features
model = PolynomialRegressionModel(input_dim).to(device)
criterion = nn.MSELoss()  # Mean Squared Error as the loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# ======================================================
# 5. Train the Model
# ======================================================
# Train the model for 100 epochs
train_loss_values = []  # To store training loss values

for epoch in range(100):
    model.train()  # Set the model to training mode

    # Forward pass
    train_outputs = model(X_train_scaled)
    train_loss = criterion(train_outputs.flatten(), y_train_scaled.flatten())

    # Backward pass and optimization
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Log training loss
    train_loss_values.append(train_loss.item())

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Training Loss: {train_loss.item():.4f}")

# Save the trained model weights
torch.save(model, "models/PLR_weights.pth")

# ======================================================
# 6. Plot Training Loss
# ======================================================
# Visualize the training loss over epochs
plt.plot(range(1, 101), train_loss_values, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("../../reports/figures/PLR_loss.png")  # Save plot to a file
plt.close()

# ======================================================
# 7. Evaluate the Model
# ======================================================
# Evaluate the model on the test set
with torch.no_grad():  # Disable gradient computation
    model.eval()  # Set the model to evaluation mode
    y_pred_scaled = model(X_test_scaled).flatten()

# Calculate Mean Squared Error (MSE) on the test set
mse = np.mean(
    (y_pred_scaled.cpu().numpy() - y_test_scaled.cpu().numpy().flatten()) ** 2
)
print(f"Test MSE: {mse:.4f}")

# ======================================================
# 8. Statistical Analysis (Optional)
# ======================================================
# Perform statistical analysis using OLS (Ordinary Least Squares) from statsmodels
X_train_ols = sm.add_constant(X_train_poly)  # Add intercept term
ols_model = sm.OLS(y_train, X_train_ols).fit()

# Save the OLS summary to a file
with open("../../reports/PLR_ols_summary.txt", "w") as f:
    f.write(f"Test MSE: {mse:.4f}\n")
    f.write(ols_model.summary().as_text())

# ======================================================
# 9. Save Preprocessing Objects
# ======================================================
# Save the scalers and polynomial feature transformer for future use
joblib.dump(scaler_features, "models/PLR_scaler_X.pkl")
joblib.dump(scaler_target, "models/PLR_scaler_y.pkl")
joblib.dump(poly_reg, "models/poly.pkl")
