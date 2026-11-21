
# MARKETMIND AI
## Real Market Sales Analysis Using Machine Learning

### Project Overview

MARKETMIND AI is a machine learning project designed
to analyze sales data and customer purchasing behavior.

The application contains two independent AI workflows:
supplier sales prediction and customer segmentation.

### Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib

### Machine Learning Models

#### 1. Supplier Sales Prediction

Algorithm: Random Forest Regressor

The model predicts sales-related values using supplier
information and other input features.

Saved evaluation results:

- R² Score: 0.9784
- MAE: 31.89
- RMSE: 96.67

#### 2. Customer Segmentation

The model analyzes customers using RFM features:

- Recency
- Frequency
- Monetary Value

Customers are grouped into four segments based
on purchasing behavior.

### Application Modules

1. Executive Overview
2. Supplier Sales AI
3. Customer Segmentation AI
4. Model Performance
5. Prediction Center

### How to Run

Install dependencies:

    pip install -r requirements.txt

Start the application from the project root:

    python -m streamlit run app.py/app.py

### Project Author

Prashant Prajapati

B.Tech CSE (AI & Data Science)

Shri Ramswaroop Memorial University, Lucknow
