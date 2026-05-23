import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = os.path.join('day9', 'day9_cleaning.ipynb')

nb = nbf.v4.new_notebook()

# Define the cells for the Jupyter notebook
cells = [
    nbf.v4.new_markdown_cell("""# Day 9: Data Cleaning & Preparation

In this notebook, I'll clean a messy real-world transaction dataset. Real-world datasets often have missing values, duplicate entries, inconsistent formats, and wrong data types. 

I will go through:
1. Finding and removing duplicates (exact and key-based).
2. Standardizing text columns (segment names, categories, and country formats).
3. Fixing inconsistent date formats.
4. Cleaning numerical columns (removing currency symbols from sales, dealing with negative values).
5. Formatting postal codes (adding back leading zeros).
6. Imputing missing values with statistical methods (median imputation for sales, mode for quantities)."""),

    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Load messy transactions
df = pd.read_csv('dirty_store_transactions.csv')
print(f"Initial shape: {df.shape}")
df.head(10)"""),

    nbf.v4.new_markdown_cell("""### 1. Identify Missing Values and Data Types
Let's inspect the column types and count missing values in each column."""),

    nbf.v4.new_code_cell("""df.info()"""),

    nbf.v4.new_code_cell("""print("Missing values per column:")
df.isnull().sum()"""),

    nbf.v4.new_markdown_cell("""### 2. Handle Duplicate Records
Let's check for duplicate records. We want to check for exact duplicates first, then logical duplicates (where the same transaction details are logged, possibly under a different Row ID)."""),

    nbf.v4.new_code_cell("""# Check for exact duplicate rows
exact_dupes = df.duplicated()
print(f"Exact duplicates found: {exact_dupes.sum()}")

# Check duplicates ignoring 'Row ID'
logical_dupes = df.duplicated(subset=[col for col in df.columns if col != 'Row ID'])
print(f"Logical duplicates (ignoring Row ID): {logical_dupes.sum()}")"""),

    nbf.v4.new_code_cell("""# Let's drop exact duplicates first
df = df.drop_duplicates()

# Now drop logical duplicates, keeping the first entry
df = df.drop_duplicates(subset=[col for col in df.columns if col != 'Row ID'], keep='first')
print(f"Shape after removing duplicates: {df.shape}")"""),

    nbf.v4.new_markdown_cell("""### 3. Clean Text Columns
Text fields like Customer Name, Segment, Category, and Country have casing inconsistencies and typos. Let's inspect them."""),

    nbf.v4.new_code_cell("""print("Unique segments:", df['Segment'].unique())
print("Unique categories:", df['Category'].unique())
print("Unique countries:", df['Country'].unique())"""),

    nbf.v4.new_code_cell("""# Customer Name: strip whitespace, title case, and fill missing with 'Unknown'
df['Customer Name'] = df['Customer Name'].str.strip().str.title()
df['Customer Name'] = df['Customer Name'].fillna('Unknown')

# Category: strip whitespace, title case
df['Category'] = df['Category'].str.strip().str.title()

# Segment: clean casing and specific typo 'consumer_typo'
df['Segment'] = df['Segment'].str.strip().str.title()
df['Segment'] = df['Segment'].replace('Consumer_Typo', 'Consumer')
df['Segment'] = df['Segment'].fillna('Consumer') # fill missing with majority class

# Country: standardize to 'United States'
country_map = {
    'US': 'United States',
    'USA': 'United States',
    'United States Of America': 'United States',
    'United States': 'United States'
}
df['Country'] = df['Country'].str.strip().str.title()
df['Country'] = df['Country'].map(country_map).fillna('United States')

# Verify changes
print("Cleaned segments:", df['Segment'].unique())
print("Cleaned categories:", df['Category'].unique())
print("Cleaned countries:", df['Country'].unique())"""),

    nbf.v4.new_markdown_cell("""### 4. Parse Dates
Dates are stored in different formats and some are invalid. Let's parse them using pandas `to_datetime` with `errors='coerce'` to flag invalid dates as NaNs."""),

    nbf.v4.new_code_cell("""# Parse dates using format='mixed' and coerce to handle invalid dates
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', errors='coerce')
print("Missing dates after parsing:", df['Order Date'].isnull().sum())"""),

    nbf.v4.new_code_cell("""# Show rows where dates are null (the invalid/missing dates)
df[df['Order Date'].isnull()]"""),

    nbf.v4.new_code_cell("""# Since order date is critical, drop rows where date is missing
df = df.dropna(subset=['Order Date'])
print(f"Shape after dropping missing dates: {df.shape}")"""),

    nbf.v4.new_markdown_cell("""### 5. Clean Sales Column
Sales is currently a string because of currency symbols (`$`), commas, and units like `"USD"`. It also contains negative outliers like `-999`. Let's clean and convert it to numeric."""),

    nbf.v4.new_code_cell("""# Remove currency symbols, commas, spaces and text
df['Sales'] = df['Sales'].astype(str)
df['Sales'] = df['Sales'].str.replace('$', '', regex=False)
df['Sales'] = df['Sales'].str.replace(',', '', regex=False)
df['Sales'] = df['Sales'].str.replace(' USD', '', regex=False)
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')

# Check distribution of sales
df['Sales'].describe()"""),

    nbf.v4.new_markdown_cell("""We have some negative sales (like -999.0) and missing sales values. Let's set negative values to NaN and impute all NaNs using the median sales value of their product category."""),

    nbf.v4.new_code_cell("""# Treat negative sales as missing values
df.loc[df['Sales'] < 0, 'Sales'] = np.nan

# Impute missing sales with the median sales of their product category
df['Sales'] = df.groupby('Category')['Sales'].transform(lambda x: x.fillna(x.median()))
print("Missing sales remaining:", df['Sales'].isnull().sum())
df['Sales'].describe()"""),

    nbf.v4.new_markdown_cell("""### 6. Clean Quantity Column
Let's clean and convert the quantity column to integer, filling missing values with the median quantity (which is typically a safe default for count data)."""),

    nbf.v4.new_code_cell("""# Convert to numeric first
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

# Fill missing quantities with median
median_qty = df['Quantity'].median()
df['Quantity'] = df['Quantity'].fillna(median_qty).astype(int)

print("Quantity summary:")
df['Quantity'].value_counts()"""),

    nbf.v4.new_markdown_cell("""### 7. Standardize Postal Codes
Postal codes are read as floats or strings of varying lengths. Let's convert them to 5-digit strings, prefixing leading zeros if they were lost."""),

    nbf.v4.new_code_cell("""def clean_postal_code(val):
    if pd.isnull(val):
        return 'Unknown'
    # Convert float representation to string, stripping decimals
    val_str = str(val).split('.')[0].strip()
    if val_str == 'nan' or val_str == '':
        return 'Unknown'
    # Pad with leading zeros up to 5 characters
    return val_str.zfill(5)

df['Postal Code'] = df['Postal Code'].apply(clean_postal_code)
print("Unique postal codes preview:")
df['Postal Code'].value_counts().head(10)"""),

    nbf.v4.new_markdown_cell("""### 8. Final Inspection & Export
Let's view the clean dataset info, make sure there are no nulls left, and save it to a CSV file."""),

    nbf.v4.new_code_cell("""df.info()"""),

    nbf.v4.new_code_cell("""df.head(10)"""),

    nbf.v4.new_code_cell("""# Save clean dataset
df.to_csv('cleaned_store_transactions.csv', index=False)
print("Cleaned dataset successfully exported!")""")
]

nb['cells'] = cells

# Write notebook file
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created at {notebook_path}. Executing now...")

# Run notebook
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': 'day9'}})

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_to_run, f)

print("Notebook run complete and saved with outputs!")
