# 🚀 Retail Sales Analysis, Profit Prediction & Sales Forecasting

## 📌 Project Overview

This project demonstrates an end-to-end Data Science workflow using retail business data. Multiple datasets including customers, products, locations, and orders were integrated to generate business insights through Exploratory Data Analysis (EDA), Data Visualization, Machine Learning, Customer Segmentation, and Time Series Forecasting.

The project helps businesses understand sales performance, customer behaviour, profitability, and future sales trends for data-driven decision making.

## 🎯 Objectives
Clean and preprocess retail datasets
Merge multiple datasets into a single analytical dataset
Perform Exploratory Data Analysis (EDA)
Visualize sales and profit trends
Identify top-performing customers, products, and regions
Predict profit using Machine Learning models
Classify customer segments
Forecast future sales using ARIMA Time Series Model

## 📂 Dataset Information

The project uses four datasets:

1. Customers Dataset

Contains:

Customer ID
Customer Name
Segment
2. Products Dataset

Contains:

Product ID
Category
Sub-Category
3. Locations Dataset

Contains:

Postal Code
City
State
Region
4. Orders Dataset

Contains:

Order ID
Order Date
Ship Date
Sales
Profit
Discount
Quantity
Ship Mode

## 🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
StatsModels
VS Code

## 📈 Exploratory Data Analysis

The following business questions were answered:

✅ Total Sales Analysis

✅ Total Profit Analysis

✅ Sales by Region

✅ Sales by Category

✅ Top 10 Customers by Sales

✅ Profit vs Sales Relationship

✅ Monthly Sales Trend Analysis

✅ Discount Impact on Profit

✅ Segment-wise Sales Analysis

✅ Product Sub-Category Profitability Analysis

## 📊 Visualizations

This project includes:

Sales by Region Bar Chart
Category-wise Sales Pie Chart
Top Customers Horizontal Bar Chart
Profit vs Sales Scatter Plot
Monthly Sales Trend
Discount vs Profit Analysis
Dashboard-Style Summary Visualization

## 🤖 Machine Learning Models
1. Linear Regression

Predicting Profit using:

Sales
Discount
Quantity

Evaluation Metrics:

R² Score
RMSE
2. Random Forest Regression

Improved profit prediction using:

Sales
Discount
Quantity

Features analysed through Feature Importance Scores.

3. Random Forest Regression with Categorical Features

Additional Features:

Category
Region

Techniques Used:

OneHot Encoding
Pipeline
ColumnTransformer

## 🎯 Customer Segment Classification
Model Used:

Random Forest Classifier

Target Variable:

Segment

Features:

Sales
Discount
Quantity
Category
Region

Evaluation:

Accuracy Score
Precision
Recall
F1 Score
Confusion Matrix

## 📉 Time Series Forecasting
ARIMA Model

Used ARIMA (1,1,1) model to forecast future sales.

Forecasting Steps
Convert Order Date to Datetime
Aggregate Monthly Sales
Train ARIMA Model
Forecast Next 12 Months
Visualize Forecasted Sales

## Business Applications:
Demand Forecasting
Inventory Planning
Budget Estimation
Strategic Business Planning

## 🔍 Key Insights
Certain regions contribute significantly more revenue.
A small percentage of customers generate a large share of total sales.
High discounts often reduce profitability.
Some product sub-categories are loss-making despite strong sales.
Random Forest performed better than Linear Regression.
Historical sales trends can be effectively used for forecasting future demand.

## 📸 Sample Outputs

Some screenshots of Project:

Sales by Region Chart
<img width="795" height="652" alt="Screenshot 5-BarChart" src="https://github.com/user-attachments/assets/d96feeef-5323-4945-b4ac-af762b012211" />

Profit Prediction Results
<img width="640" height="480" alt="PredictiveVSActualProft(RanFor)" src="https://github.com/user-attachments/assets/cf0b57e3-62fa-4f7f-94ea-cd35c69f9517" />

Classification Results
<img width="640" height="480" alt="ImprovedClassification" src="https://github.com/user-attachments/assets/de368087-7e4a-4e8b-8abc-86b3db9b3cfa" />

ARIMA Forecast Graph
<img width="1200" height="600" alt="SalesForecast(next 12 months)" src="https://github.com/user-attachments/assets/d1b3c790-f8b1-47a6-a353-f6645b823314" />


Example:

Markdown
Dashboard:
<img width="1536" height="754" alt="Screenshot 7-DashboardStyleVisualization" src="https://github.com/user-attachments/assets/936274cd-d45d-4b63-8c1f-492856df900c" />

 
Sales Forcast:
<img width="1200" height="600" alt="SalesForecast(next 12 months)" src="https://github.com/user-attachments/assets/77907d6f-ba35-4788-b52f-1580ce20936a" />

📁 Project Structure
Plain Text
Retail-Sales-Analysis/
│
├── Data/
│ ├── Customers.csv
│ ├── Products.csv
│ ├── Location.csv
│ └── Orders.csv
│
├── Notebook/
│ └── Retail_Sales_Analysis.ipynb
│
├── Images/
│ ├── dashboard.png
│ └── forecast.png
│
├── Report/
│ └── Project_Report.pdf
│
├── README.md
│
└── requirements.txt

# 🚀 Future Improvements
Power BI Dashboard
Tableau Dashboard
Customer Churn Prediction
Product Recommendation System
Prophet Forecasting

# 👨‍💻 Author
Varsha K
Aspiring Data Scientist | Python | Pandas | Machine Learning | Data Analytics

GitHub: https://github.com/Varsha-K22
LinkedIn: https://linkedin.com/in/your-profile](https://www.linkedin.com/in/varsha-kutumbale-6b0b28101

# ⭐ Project Highlights

✔ Complete Data Science Workflow

✔ Data Cleaning & Integration

✔ Exploratory Data Analysis

✔ Business Intelligence Insights

✔ Machine Learning Models

✔ Customer Classification

✔ Time Series Forecasting using ARIMA

✔ Industry-Oriented Retail Analytics Project
