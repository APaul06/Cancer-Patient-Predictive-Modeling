import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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
print("First 5 predictions:", y_pred[:5])

