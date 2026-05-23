# Day 6: Data Cleaning & Preprocessing

## Objective
The goal for today is to practice data cleaning and preprocessing techniques, recognizing that real-world data is often messy and incomplete. 

## What I Did:
- **Handled missing values**: Filled missing numerical values with medians and categorical values with modes.
- **Removed duplicates**: Identified and dropped duplicate rows in the dataset.
- **Fixed data types**: Converted `Sales` and `Quantity` to numeric, and `Order Date` and `Ship Date` to datetime formats.
- **Created 2 new features**: 
  - `Processing Time`: The time taken from order to shipment (in days).
  - `Sales per Item`: Calculated by dividing `Sales` by `Quantity`.

## Files
- `messy_dataset.csv`: The initial generated dataset with missing values, incorrect types, and duplicates.
- `day6_data_cleaning.ipynb`: The Jupyter Notebook containing the data cleaning code and process.
- `cleaned_dataset.csv`: The final preprocessed and clean dataset ready for analysis.
