
# STEP - I - Import the libraries, dataset, and analysing
# Step 1.1 - Import Libraries
import pandas as pd


# Step 1.2 - Load datasets
customers = pd.read_csv("Customers.csv", sep=";", encoding="latin1")
locations = pd.read_csv("Location.csv", sep=";", encoding="latin1")
products = pd.read_csv("Products.csv", sep=";", encoding="latin1")

# Orders dataset is too large, so we’ll handle it carefully later
# For now, just try loading it with low_memory=False
orders = pd.read_csv("Orders.csv", sep=";", low_memory=False)

# Step 1.3 - Inspect the first few rows
print("Customer Data:\n", customers.head())
print("\n")   
print("Location Data: \n", locations.head())
print("\n")   
print("Product Data: \n", products.head())
print("\n")   
print("Orders Data: \n", orders.head())

# STEP - II - Explore and Clean the data
# Step 2.1 - Explore data
# Customers
print("Customer Info: \n")
print(customers.info())
print("\n")
print("Missing Values: \n")
print(customers.isnull().sum())
print("="*50)

# Location
print("Location Info: \n")
print(locations.info())
print("\n")
print("Missing Values: \n")
print(locations.isnull().sum())
print("="*50)

# Products
print("Product Info: \n")
print(products.info())
print("\n")
print("Missing Values: \n")
print(products.isnull().sum())
print("="*50)

# Orders
print("Orders Info: \n")
print(orders.info())
print("\n")
print("Missing Values: \n")
print(orders.isnull().sum())
print("="*50)

# STEP - III - Merge Dataset Together
# Merge Orders with Customers
orders_customers = pd.merge(orders, customers, on="Customer ID", how="left")

# Merge with Products
orders_products = pd.merge(orders_customers, products, on="Product ID", how="left")

# Merge with Locations (using Postal Code)
final_data = pd.merge(orders_products, locations, on="Postal Code", how="left")

# Inspect Final Merge Dataset
print("Final Merged Dataset")
print(final_data.head())
print(final_data.info())

# STEP - IV - BASIC EDA
# Step 4.1 - Overall Sales and Profit
print("Total Sales", final_data['Sales'].sum())
print("Total Profit", final_data['Profit'].sum())

# Step 4.2 - Sales by Region
print("\nSales by Region")
print(final_data.groupby('Region')['Sales'].sum().sort_values(ascending=False))

# # Step 4.3 - Sales by Category
print("\nSales by Category")
print(final_data.groupby('Category')['Sales'].sum().sort_values(ascending=False))

# # Step 4.4 - Top 10 Customers by Sales
print("\n Top 10 Customers by Sales")
print(final_data.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10))

# STEP - V - VISUALIZE THE INSIGHTS
import seaborn as sns
import matplotlib.pyplot as plt

# Step 5.1 - Sales by Region (Bar Chart)
region_sales = final_data.groupby('Region')['Sales'].sum().sort_values(ascending=False)
sns.barplot(x=region_sales.index, y=region_sales.values)
plt.title("Sales by Region")
plt.ylabel("Total Sales")
# plt.show()

# # Step 5.2 - Sales by Category (Pie Chart)
category_sales = final_data.groupby('Category')['Sales'].sum()
plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%')
plt.title("Sales Distribution by Category")
# plt.show()

# # Step 5.3 - Top 10 Customers (Horizontal Bar Chart)
top_customers = final_data.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
sns.barplot(y=top_customers.index, x=top_customers.values)
plt.title("Top 10 Customers by Sales")
plt.xlabel("Total Sales")
plt.show()

# # Step 5.4 - Profit vs Sales Scatter Plot (to see which orders are high sales but low profit)
sns.scatterplot(data=final_data, x='Sales', y='Profit', alpha=0.6)
plt.title("Profit VS Sales")
plt.show()

# STEP 6 - Next-Level Analysis

# Convert Order Date column to datetime
final_data['Order Date'] = pd.to_datetime(final_data['Order Date'], dayfirst=True, errors='coerce')

# Convert Ship Date column too (important for shipping analysis later)
final_data['Ship Date'] = pd.to_datetime(final_data['Ship Date'], dayfirst=True, errors='coerce')

# Check conversion
print(final_data[['Order Date', 'Ship Date']].head())

# Step 6.1 - Time Series Analysis (Look at sales trends over months/years.)
final_data['Order Date'] = pd.to_datetime(final_data['Order Date'])
monthly_sales = final_data.groupby(final_data['Order Date'].dt.to_period("M"))['Sales'].sum()
monthly_sales.plot(kind='line', figsize=(12,6), title="Monthly Sales Trend")
plt.show()

# Step 6.2 - Profitability by Product Sub-Category (Identify which products are profitable vs loss-making.)
subcat_profit = final_data.groupby('Sub-Category')['Profit'].sum().sort_values()
subcat_profit.plot(kind="barh", figsize=(10,6), title="Profit by Subcategory")
plt.show()

# Step 6.3 - Customer Segmentation (Compare sales/profit across segments (Consumer, Corporate, Home Office).)
segment_sales = final_data.groupby('Segment')['Sales'].sum()
segment_sales.plot(kind='bar', title="Sales by Segment")
plt.show()

# Step 6.4 - Discount Impact on Profit (See how discounts affect profitability.)
sns.scatterplot(data=final_data, x='Discount', y='Profit', alpha=0.6)
plt.title("Discount vs Profit")
plt.show()


# # STEP 7 - DASHBOARD STYLE VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14,10))

# # Sales by Region
region_sales = final_data.groupby('Region')['Sales'].sum().sort_values(ascending=False)
sns.barplot(x=region_sales.index, y=region_sales.values, ax=axes[0,0])
axes[0,0].set_title("Sales by Region")

# # Sales by Category
category_sales = final_data.groupby('Category')['Sales'].sum()
axes[0,1].pie(category_sales, labels=category_sales.index, autopct='%1.1f%%')
axes[0,1].set_title("Sales Distribution by Category")

# # Top 10 Customers
top_customers = final_data.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
sns.barplot(y=top_customers.index, x=top_customers.values, ax=axes[1,0])
axes[1,0].set_title("Top 10 Customers by Sales")

# # Profit vs Sales
sns.scatterplot(data=final_data, x='Sales', y='Profit', alpha=0.6, ax=axes[1,1])
axes[1,1].set_title("Profit vs Sales")

plt.tight_layout()
plt.show()


# STEP - 8 PREDECTIVE ANALYSIS - Machine Learning Model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# Step 8.1 - Prepare the data
# Select feature and target
x = final_data[['Sales','Discount', 'Quantity']]
y = final_data['Profit']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 8.2 - Train a Model
# Train Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)


# Convert to arrays
y_test_array = y_test.values
y_pred_array = y_pred

# Evaluate
print("Linear Regression Result: \n")
print("R2 Score: ", r2_score(y_test_array, y_pred_array))
print("RMSE", root_mean_squared_error(y_test_array, y_pred_array))

# Step 8.3 - Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

print("\n🌲 Random Forest Results")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("RMSE:", root_mean_squared_error(y_test, y_pred_rf))

# # Step 8.4 - Feature Importance (Random Forest)
importances = rf_model.feature_importances_
features = x.columns

plt.bar(features, importances)
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance Score")
plt.show()

# Step 8.5: Predicted vs Actual Plot (Random Forest)
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Predicted vs Actual Profit (Random Forest)")
plt.show()

# STEP 9 - # Step 4: Regression with Categorical Features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
# Features: numeric + categorical
X = final_data[['Sales', 'Discount', 'Quantity', 'Category', 'Region']]
y = final_data['Profit']

# # Preprocessing: One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'Region'])
    ],
    remainder='passthrough'  # keep numeric features
)

# # Random Forest pipeline
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("🌲 Random Forest with Categorical Features")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("RMSE:", root_mean_squared_error(y_test, y_pred_rf))

# STEP 10 - Classification Model (Predict Customer Segment)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Features and target
X_class = final_data[['Sales', 'Discount', 'Quantity', 'Category', 'Region']]
y_class = final_data['Segment']

# Preprocessing pipeline (same as regression)
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Fit model
clf_pipeline.fit(X_train, y_train)
y_pred_class = clf_pipeline.predict(X_test)

print("Classification Results")
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))

# STEP 11 - IMPROVE CLASSIFICATION
X_class = final_data[['Sales', 'Discount', 'Quantity', 'Category', 'Region', 'Ship Mode']]
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
])
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=y_test.unique(),
            yticklabels=y_test.unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# STEP 12 - Time Series Forecasting
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Convert Order Date to datetime
final_data['Order Date'] = pd.to_datetime(final_data['Order Date'], dayfirst=True, errors='coerce')

# Step 2: Aggregate monthly sales
monthly_sales = final_data.groupby(final_data['Order Date'].dt.to_period("M"))['Sales'].sum()
monthly_sales = monthly_sales.to_timestamp()

# Step 3: Fit ARIMA model
model = sm.tsa.ARIMA(monthly_sales, order=(1,1,1))  # (p,d,q) parameters
results = model.fit()

# Step 4: Forecast next 12 months
forecast = results.forecast(steps=12)

# Step 5: Plot
plt.figure(figsize=(12,6))
plt.plot(monthly_sales, label="Historical Sales")
plt.plot(forecast.index, forecast, label="Forecast", color="red")
plt.title("Sales Forecast (Next 12 Months)")
plt.legend()
plt.show()
