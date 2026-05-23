import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(88)

# Make directory if not exists
os.makedirs('day8', exist_ok=True)

# Generate 1000 rows of sales data
n_rows = 1000

# Helper to generate dates with seasonality (peaks in Nov/Dec)
def generate_dates(n):
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2019, 12, 31)
    days_range = (end_date - start_date).days
    
    dates = []
    while len(dates) < n:
        # random days offset
        random_days = np.random.randint(0, days_range)
        date = start_date + timedelta(days=random_days)
        
        # Seasonality: increase probability for Q4 (Nov, Dec)
        if date.month in [11, 12]:
            # Keep Q4 dates with high probability
            dates.append(date)
        else:
            # Keep other dates with 50% probability to skew towards Q4
            if np.random.rand() > 0.5:
                dates.append(date)
    return sorted(dates[:n])

order_dates = generate_dates(n_rows)

# Generate Ship Dates (Order Date + 2 to 7 days)
ship_dates = []
for order_date in order_dates:
    ship_delay = np.random.randint(2, 7)
    ship_dates.append(order_date + timedelta(days=ship_delay))

# Inject shipping anomalies (student wants to spot outliers)
# Anomaly 1: Ship date BEFORE order date (system entry error)
ship_dates[15] = order_dates[15] - timedelta(days=2)
ship_dates[150] = order_dates[150] - timedelta(days=1)

# Anomaly 2: Very delayed shipping (outliers)
ship_dates[400] = order_dates[400] + timedelta(days=45)
ship_dates[750] = order_dates[750] + timedelta(days=32)

# Lists for categorical features
segments = ['Consumer', 'Corporate', 'Home Office']
segment_probs = [0.5, 0.3, 0.2]

regions = ['West', 'East', 'Central', 'South']
region_probs = [0.35, 0.30, 0.20, 0.15]

cities_by_state = {
    'California': ('Los Angeles', 'San Francisco', 'San Diego', '90036'),
    'New York': ('New York City', 'Albany', 'Buffalo', '10008'),
    'Texas': ('Houston', 'Dallas', 'Austin', '77041'),
    'Washington': ('Seattle', 'Tacoma', 'Spokane', '98103'),
    'Florida': ('Miami', 'Fort Lauderdale', 'Tampa', '33311'),
    'Illinois': ('Chicago', 'Naperville', 'Peoria', '60610')
}
states = list(cities_by_state.keys())

categories = ['Furniture', 'Office Supplies', 'Technology']
subcategories = {
    'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
    'Office Supplies': ['Paper', 'Binders', 'Art', 'Storage', 'Labels', 'Appliances'],
    'Technology': ['Phones', 'Copiers', 'Machines', 'Accessories']
}

product_names = {
    'Chairs': 'Hon Deluxe Fabric Upholstered Stacking Chairs',
    'Tables': 'Bretford CR4500 Series Slim Rectangular Table',
    'Bookcases': 'Bush Somerset Collection Bookcase',
    'Furnishings': 'Eldon Expressions Desk Organizer',
    'Paper': 'Xerox 1970 Multipurpose Paper',
    'Binders': 'Ibico Standard Binders',
    'Art': 'Newell 318 Art Markers',
    'Storage': 'Eldon Fold \'N Roll Cart System',
    'Labels': 'Self-Adhesive Address Labels',
    'Appliances': 'Kensington SafeMouse Mouse Pad with Gel Wrist Rest',
    'Phones': 'Samsung Galaxy S4',
    'Copiers': 'Hewlett Packard LaserJet 3310 Copier',
    'Machines': 'Lexmark MX611dhe Monochrome Laser Printer',
    'Accessories': 'Logitech Wireless Headset'
}

# Customer Data
customer_names = [
    'Claire Gute', 'Darrin Van Huff', 'Sean O\'Donnell', 'Brosina Hoffman', 
    'Zuschuss Donatelli', 'Ruben Hope', 'Ken Lonsdale', 'Sandra Flanagan', 
    'Irene Maddox', 'Harold Pawlan', 'Pete Kriz', 'Alejandro Grove'
]

# Generate rows
rows = []
for i in range(n_rows):
    row_id = i + 1
    order_id = f"CA-201{np.random.randint(6,9)}-{100000 + np.random.randint(10000, 99999)}"
    
    order_d = order_dates[i]
    ship_d = ship_dates[i]
    
    ship_mode = np.random.choice(['Standard Class', 'Second Class', 'First Class', 'Same Day'], p=[0.6, 0.2, 0.15, 0.05])
    
    cust_name = np.random.choice(customer_names)
    cust_id = cust_name.replace(" ", "")[:2].upper() + f"-{np.random.randint(10000, 99999)}"
    
    segment = np.random.choice(segments, p=segment_probs)
    state = np.random.choice(states)
    city_info = cities_by_state[state]
    city = city_info[np.random.randint(0, 3)]
    postal_code = city_info[3]
    region = np.random.choice(regions, p=region_probs)
    
    category = np.random.choice(categories)
    subcat = np.random.choice(subcategories[category])
    prod_name = product_names[subcat]
    prod_id = f"{category[:3].upper()}-{subcat[:2].upper()}-{10000000 + np.random.randint(1000, 9999)}"
    
    # Generate Sales based on Category
    if category == 'Technology':
        sales = np.random.normal(loc=450, scale=150)
        sales = max(15.0, sales)
    elif category == 'Furniture':
        sales = np.random.normal(loc=280, scale=90)
        sales = max(10.0, sales)
    else: # Office Supplies
        sales = np.random.normal(loc=35, scale=15)
        sales = max(2.5, sales)
        
    # Introduce some variation based on segment
    if segment == 'Corporate':
        sales *= 1.15  # corporate orders are slightly larger
    elif segment == 'Home Office':
        sales *= 1.05
        
    sales = round(sales, 2)
    
    rows.append({
        'Row ID': row_id,
        'Order ID': order_id,
        'Order Date': order_d.strftime('%m/%d/%Y'),
        'Ship Date': ship_d.strftime('%m/%d/%Y'),
        'Ship Mode': ship_mode,
        'Customer ID': cust_id,
        'Customer Name': cust_name,
        'Segment': segment,
        'Country': 'United States',
        'City': city,
        'State': state,
        'Postal Code': postal_code,
        'Region': region,
        'Product ID': prod_id,
        'Category': category,
        'Sub-Category': subcat,
        'Product Name': prod_name,
        'Sales': sales
    })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Inject high outlier sales anomalies
# Technology / Copier
df.loc[120, 'Sales'] = 22638.48
df.loc[120, 'Product Name'] = 'Canon imageCLASS 2200 Advanced Copier'
df.loc[120, 'Sub-Category'] = 'Copiers'
df.loc[120, 'Category'] = 'Technology'

# Machine
df.loc[530, 'Sales'] = 9099.90
df.loc[530, 'Product Name'] = '3D Systems Cube Printer'
df.loc[530, 'Sub-Category'] = 'Machines'
df.loc[530, 'Category'] = 'Technology'

# Furniture / Table
df.loc[840, 'Sales'] = 4500.00
df.loc[840, 'Product Name'] = 'Conference Table Custom Walnut'
df.loc[840, 'Sub-Category'] = 'Tables'
df.loc[840, 'Category'] = 'Furniture'

# Write to file
output_path = os.path.join('day8', 'sales_data.csv')
df.to_csv(output_path, index=False)
print(f"Dataset successfully created at {output_path} with {len(df)} rows.")
print("Average sales by category:")
print(df.groupby('Category')['Sales'].mean())
