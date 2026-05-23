import os
import pandas as pd
import numpy as np
import nbformat as nbf

# 1. Create directory
day6_dir = 'day6'
os.makedirs(day6_dir, exist_ok=True)

# 2. Generate messy dataset
data = {
    'Row ID': [1, 2, 3, 4, 5, 2, 6, 7, 8, 9, 10],
    'Order ID': ['CA-2016-152156', 'CA-2016-152156', 'CA-2016-138688', 'US-2015-108966', 'US-2015-108966', 'CA-2016-152156', 'CA-2017-114412', 'CA-2017-114412', 'US-2016-118983', 'US-2016-118983', 'CA-2014-105893'],
    'Order Date': ['11/8/2016', '11/8/2016', '6/12/2016', '10/11/2015', '10/11/2015', '11/8/2016', '4/15/2017', '4/15/2017', '11/22/2016', '11/22/2016', '11/11/2014'],
    'Ship Date': ['11/11/2016', '11/11/2016', '6/16/2016', '10/18/2015', '10/18/2015', '11/11/2016', '4/20/2017', '4/20/2017', '11/26/2016', '11/26/2016', '11/18/2014'],
    'Customer Name': ['Claire Gute', 'Claire Gute', 'Darrin Van Huff', 'Sean O\'Donnell', 'Sean O\'Donnell', 'Claire Gute', 'Brosina Hoffman', 'Brosina Hoffman', 'Zuschuss Donatelli', 'Zuschuss Donatelli', 'Ruben Hope'],
    'Category': ['Furniture', 'Furniture', 'Office Supplies', 'Furniture', 'Office Supplies', 'Furniture', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Office Supplies', 'Furniture'],
    'Sales': ['261.96', '731.94', '14.62', '957.5775', '22.368', '731.94', '15.552', np.nan, '68.81', '2.808', '78.85'], # Sales as string and nan
    'Quantity': ['2', '3', '2', '5', '2', '3', '3', '2', '5', '3', '4'], # Quantity as string
    'Profit': [41.91, 219.58, 6.87, -383.03, 2.51, 219.58, 5.44, 3.2, -12.04, 0.88, np.nan] # Profit with nan
}

df = pd.DataFrame(data)
# Add some missing values artificially
df.loc[2, 'Category'] = np.nan

df.to_csv(os.path.join(day6_dir, 'messy_dataset.csv'), index=False)

# 3. Create Jupyter Notebook
nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# day 6 data cleaning\ncleaning up the dataset"),
    
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('messy_dataset.csv')\ndf.head()"),
    
    nbf.v4.new_markdown_cell("### 1. missing values\nchecking for missing values"),
    
    nbf.v4.new_code_cell("df.isnull().sum()"),
    
    nbf.v4.new_code_cell("df['Category'] = df['Category'].fillna(df['Category'].mode()[0])\ndf['Profit'] = df['Profit'].fillna(df['Profit'].median())\n\ndf.isnull().sum()"),
    
    nbf.v4.new_markdown_cell("### 2. remove duplicates"),
    
    nbf.v4.new_code_cell("df.duplicated().sum()\ndf = df.drop_duplicates()\ndf.duplicated().sum()"),
    
    nbf.v4.new_markdown_cell("### 3. fix data types"),
    
    nbf.v4.new_code_cell("df.info()"),
    
    nbf.v4.new_code_cell("df['Sales'] = pd.to_numeric(df['Sales'])\ndf['Quantity'] = pd.to_numeric(df['Quantity'])\n\ndf['Order Date'] = pd.to_datetime(df['Order Date'])\ndf['Ship Date'] = pd.to_datetime(df['Ship Date'])\n\ndf.info()"),
    
    nbf.v4.new_markdown_cell("### 4. new features"),
    
    nbf.v4.new_code_cell("df['Processing Time'] = (df['Ship Date'] - df['Order Date']).dt.days\ndf['Sales per Item'] = df['Sales'] / df['Quantity']\n\ndf.head()"),
    
    nbf.v4.new_markdown_cell("### 5. save dataset"),
    
    nbf.v4.new_code_cell("df.to_csv('cleaned_dataset.csv', index=False)")
]

nb['cells'] = cells

with open(os.path.join(day6_dir, 'day6_data_cleaning.ipynb'), 'w') as f:
    nbf.write(nb, f)

print('Setup complete.')
