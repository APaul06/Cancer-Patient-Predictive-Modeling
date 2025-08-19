import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Example connection string -- customize as per your settings
engine = create_engine('postgresql://postgres:10172006@localhost:5432/Cancer Patient Data')

# Read table into pandas
df = pd.read_sql('SELECT * FROM cancerdata', engine)

""" 

# Check if DataFrame is empty or not
if df.empty:
    print("DataFrame is empty - table might not have data or query is wrong")
else:
    print("DataFrame loaded successfully")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}\n")

# Preview first few rows
print("Preview of data:")
print(df.head())

# Check columns names and data types
print("\nColumns and data types:")
print(df.dtypes) 

"""

#print(df.columns)


# Separate target and features
y = df['target_severity_score']
X = df.drop(['target_severity_score', 'patient_id'], axis=1)

# One-hot encode categorical columns before splitting
categorical_cols = ['gender', 'country_region', 'cancer_type', 'cancer_stage']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Now split the encoded dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#print("First 5 predictions:", y_pred[:5])

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Assume y_test and y_pred are your actual and predicted values from the test set
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Create Residuals plot: residuals vs predicted values

"""
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})

plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.axhline(0, color='black', linestyle='--')
plt.show()
"""


import pandas as pd

def interactive_predict(model, feature_columns, categorical_cols):
    user_input = {}

    # Features excluding patient_id and target_severity_score
    features = {
        'age': float,
        'gender': str,
        'country_region': str,
        'year': int,
        'genetic_risk': float,
        'air_pollution': float,
        'alcohol_use': float,
        'smoking': float,
        'obesity_level': float,
        'cancer_type': str,
        'cancer_stage': str,
        'treatment_cost_usd': float,
        'survival_years': float
    }

    for feature, dtype in features.items():
        while True:
            val = input(f"Please enter {feature} ({dtype.__name__}): ")
            try:
                user_input[feature] = dtype(val)
                break
            except ValueError:
                print(f"Invalid input. Please enter a {dtype.__name__} value.")

    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical columns
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Ensure input features align with training features (fill missing columns with 0)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict severity score
    prediction = model.predict(input_df)[0]

    print(f"\nPredicted Cancer Severity Score: {prediction}")

# Usage example:
categorical_cols = ['gender', 'country_region', 'cancer_type', 'cancer_stage']
feature_columns = X.columns.tolist()  # Your training features after encoding

# Then call:
interactive_predict(model, feature_columns, categorical_cols)

