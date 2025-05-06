# Importing required dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load the dataset
boston_df = pd.read_csv("/content/BostonHousing.csv")

# ----- Data Overview Functions -----

def get_shape():
    print("Number of rows:", boston_df.shape[0])
    print("Number of columns:", boston_df.shape[1])

def get_head():
    print("\nHead of the dataset:")
    print(boston_df.head())

def get_description():
    print("\nStatistical measures description:")
    print(boston_df.describe())

def check_nulls():
    print("\nMissing values in each column:")
    print(boston_df.isnull().sum())

def show_correlation_heatmap():
    correlation = boston_df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
                annot=True, annot_kws={'size': 8}, cmap='Greens')
    plt.title("Feature Correlation Heatmap")
    plt.show()

# ----- Data Preparation -----
# Split features and labels
X = boston_df.drop(columns='price', axis=1)
Y = boston_df['price']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# ----- Model Training Function -----
def train_model():
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    return model

# ----- Evaluation Functions -----
def evaluate_model(model):
    # Training performance
    train_pred = model.predict(X_train)
    train_r2 = r2_score(Y_train, train_pred)
    train_mae = mean_absolute_error(Y_train, train_pred)
    
    print("\n--- Training Data ---")
    print("R squared error :", train_r2)
    print("Mean Absolute Error :", train_mae)

    # Test performance
    test_pred = model.predict(X_test)
    test_r2 = r2_score(Y_test, test_pred)
    test_mae = mean_absolute_error(Y_test, test_pred)

    print("\n--- Testing Data ---")
    print("R squared error :", test_r2)
    print("Mean Absolute Error :", test_mae)

    # Visualize predictions
    plt.scatter(Y_train, train_pred, color='blue', label='Train')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Train: Actual vs Predicted Prices")
    plt.legend()
    plt.show()

    plt.scatter(Y_test, test_pred, color='green', label='Test')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Test: Actual vs Predicted Prices")
    plt.legend()
    plt.show()

# ----- Run the System -----
if __name__ == "__main__":
    get_shape()
    get_head()
    get_description()
    check_nulls()
    show_correlation_heatmap()

    model = train_model()
    evaluate_model(model)
