🏡 Boston-Housing-Price-Prediction  
A machine learning project using XGBoost Regressor to predict housing prices based on features like crime rate, room count, and property tax. It covers data exploration, model training, evaluation, and a visual analysis of predictions. Built with Python, Pandas, Seaborn, scikit-learn, and XGBoost.

📖 Description  
This project applies a supervised regression model to the Boston Housing dataset to predict median house prices. It demonstrates a complete ML pipeline - starting from data loading and analysis to model training, evaluation, and visualization of prediction accuracy.

📁 Dataset  
Source: Boston Housing Dataset (from a local CSV file)  
Features include:  
- CRIM: Crime rate per capita  
- RM: Average number of rooms per dwelling  
- TAX: Property tax rate  
- INDUS, LSTAT, PTRATIO, etc.  
Target label:  
- `price`: Median value of owner-occupied homes (in $1000s)

🛠️ Installation  
Install the required Python packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```
🚀 How to Run
Clone the repository:

```bash
git clone https://github.com/KingSajxxd/House-Price-Prediction-ML.git
cd Boston-Housing-Price-Prediction
```
Make sure BostonHousing.csv is present in the root directory.

Run the Python script:

```bash
python boston_price_predictor.py
```
🧠 Model Used

# XGBoost Regressor

Powerful and fast gradient boosting model

Handles overfitting with regularization

Ideal for tabular numerical data

📊 Output

R² score and MAE (Mean Absolute Error) on both training and testing sets

Visualizations comparing actual vs predicted house prices using scatter plots

Correlation heatmap to understand feature relationships

📂 Project Structure

```bash
├── BostonHousing.csv            # Dataset file  
├── boston_price_predictor.py    # Main Python script  
└── README.md                    # Project documentation  
```