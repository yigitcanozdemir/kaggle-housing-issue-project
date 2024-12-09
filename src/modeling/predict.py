# Prediction Function
# ======================================================
# 1. Import Libraries
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import sys
import os

sys.path.append("..")
import utility.plot_settings

# Check for GPU availability for prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ======================================================
# 2. Define Prediction Function
# ======================================================
def predict(X_data, y_data, poly_degree=5, device="cuda", suffix=""):

    # Load the trained model
    model = torch.load(f"models/{suffix}_weights.pth")

    # Load the scaler models for features and target
    scaler_X = joblib.load(f"models/{suffix}_scaler_X.pkl")
    scaler_y = joblib.load(f"models/{suffix}_scaler_y.pkl")

    # Scale the input data
    X_data_scaled = scaler_X.transform(X_data)
    y_data_scaled = scaler_y.transform(y_data.values.reshape(-1, 1))

    # Process based on model type
    if suffix == "MLR":  # Preprocess for MLR model
        # Filter the features selected by RFE
        RFE_SELECTED_FEATURES = [
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "parking",
            "mainroad",
            "guestroom",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
        ]
        X_data_scaled = pd.DataFrame(X_data_scaled, columns=X_data.columns)
        X_data_scaled = X_data_scaled[RFE_SELECTED_FEATURES]
        X_data_scaled = X_data_scaled.values

    elif suffix == "PLR":  # Preprocess for PLR model
        poly_reg = joblib.load("models/poly.pkl")
        X_data_scaled = poly_reg.transform(X_data_scaled)

    else:
        raise ValueError("Unsupported model type! Only 'MLR' or 'PLR' are allowed.")

    # Perform prediction using the model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        model = model.to(device)  # Move model to device (default=GPU)
        X_data_tensor = torch.tensor(X_data_scaled, dtype=torch.float32).to(
            device
        )  # Move input data to device (default=GPU)
        pred = model(X_data_tensor)

    # Calculate Mean Squared Error (MSE) on the test set
    mse = np.mean((pred.cpu().numpy().flatten() - y_data_scaled.flatten()) ** 2)
    print(f"MSE: {mse:.4f}")

    # Inverse transform the scaled predictions
    pred = scaler_y.inverse_transform(pred.cpu().numpy().reshape(-1, 1))

    # Plotting the results
    y_true = y_data.values  # True values (unscaled)
    y_pred = pred.flatten()  # Predicted values

    # Output directory for saving the figure
    output_dir = "../../reports/figures"

    # Plot the true vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, label="Predicted vs Actual", color="blue", alpha=0.6)

    # Plot the line where error is 0 (y = x)
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        color="red",
        label="Error = 0 (y=x)",
        linestyle="--",
    )

    # Set plot labels and title
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title(f"True vs Predicted Values ({suffix})")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f"{output_dir}/{suffix}_predicted.png")
    plt.close()

    print(
        f"Model predictions and visualization completed. Figure saved to {output_dir}/{suffix}_predicted.png"
    )
    return pred


# ======================================================
# 3. Load Data and Perform Prediction
# ======================================================
# Load the data
data = pd.read_pickle("../../data/interim/housing.pkl")
df = pd.DataFrame(data)
y_test = data.pop("price")
X_test = data

# Perform prediction
predict(X_test, y_test, suffix="MLR")  # PLR/MLR
