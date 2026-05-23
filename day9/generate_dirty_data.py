import os
import pandas as pd
import numpy as np
import random

# Make directory if not exists
os.makedirs('day9', exist_ok=True)

# Set seed for reproducible messiness
np.random.seed(42)
random.seed(42)

n_rows = 1000

# Sample pool for generating records
customers = ['Claire Gute', 'Darrin Van Huff', 'Sean O\'Donnell', 'Brosina Hoffman', 
             'Zuschuss Donatelli', 'Ruben Hope', 'Ken Lonsdale', 'Sandra Flanagan']
segments = ['Consumer', 'Corporate', 'Home Office']
categories = ['Furniture', 'Office Supplies', 'Technology']
countries = ['United States', 'US', 'USA', 'United States of America']

# Generate date samples in multiple formats
date_formats = [
    '%Y-%m-%d',
    '%m/%d/%Y',
    '%d-%b-%y'
]

rows = []
for i in range(n_rows):
    row_id = i + 1
    order_id = f"CA-201{random.randint(6,9)}-{random.randint(100000, 199999)}"
    
    # Inconsistent dates
    base_date = pd.Timestamp('2016-01-01') + pd.to_timedelta(random.randint(0, 1400), unit='D')
    fmt = random.choice(date_formats)
    order_date = base_date.strftime(fmt)
    
    # Introduce random timestamp string or invalid dates
    if random.random() < 0.03:
        order_date = "2018/13/45" # invalid date
    elif random.random() < 0.02:
        order_date = np.nan
        
    # Inconsistent customer names (spaces, casing)
    cust = random.choice(customers)
    if random.random() < 0.15:
        cust = f"  {cust.lower()}  " if random.random() < 0.5 else cust.upper()
    elif random.random() < 0.05:
        cust = None
        
    # Segments with casing issues and typos
    seg = random.choice(segments)
    if random.random() < 0.1:
        seg = seg.lower()
    elif random.random() < 0.03:
        seg = "Consumer_typo"
    elif random.random() < 0.05:
        seg = np.nan
        
    # Inconsistent country representations
    country = random.choice(countries)
    if random.random() < 0.05:
        country = None
        
    # Postal codes (some missing, some as floats, some as short strings missing leading zeros)
    zip_code = random.choice(['90036', '10008', '02108', '77041', '60610'])
    if random.random() < 0.1:
        zip_code = float(zip_code) # converts to float e.g. 90036.0 or 2108.0
    elif random.random() < 0.05:
        zip_code = '02108' # keep string with leading zero
    elif random.random() < 0.08:
        zip_code = '2108' # string missing leading zero
    elif random.random() < 0.05:
        zip_code = np.nan
        
    category = random.choice(categories)
    if random.random() < 0.1:
        category = category.lower()
    
    # Inconsistent Sales format (currency symbols, commas, strings)
    base_sales = round(random.uniform(5.0, 1500.0), 2)
    sales = base_sales
    if random.random() < 0.2:
        sales = f"${base_sales:,.2f}"
    elif random.random() < 0.05:
        sales = f"{base_sales} USD"
    elif random.random() < 0.03:
        sales = "-999.0" # negative sales outlier
    elif random.random() < 0.05:
        sales = np.nan
        
    # Quantity as floats or missing
    qty = random.randint(1, 10)
    if random.random() < 0.1:
        qty = float(qty)
    elif random.random() < 0.05:
        qty = None
        
    rows.append({
        'Row ID': row_id,
        'Order ID': order_id,
        'Order Date': order_date,
        'Customer Name': cust,
        'Segment': seg,
        'Country': country,
        'Postal Code': zip_code,
        'Category': category,
        'Sales': sales,
        'Quantity': qty
    })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Inject explicit duplicate records
duplicates = df.sample(n=25, random_state=42).copy()
# Change row ID for some of them to simulate entry errors under a new ID
duplicates.iloc[:10, 0] = duplicates.iloc[:10, 0] + 10000
df = pd.concat([df, duplicates], ignore_index=True)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Save messy data
df.to_csv('day9/dirty_store_transactions.csv', index=False)
print(f"Messy dataset generated successfully with {len(df)} rows.")
