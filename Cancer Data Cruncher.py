import pandas as pd
from sqlalchemy import create_engine

# Example connection string -- customize as per your settings
engine = create_engine('postgresql://postgres:10172006@localhost:5432/Cancer Patient Data')

# Read table into pandas
df = pd.read_sql('SELECT * FROM cancerdata', engine)

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