# Day 9: Cleaning Messy Real-World Data

Real-world datasets are rarely clean. They often contain missing values, duplicate entries, inconsistent formats, casing differences, and out-of-bound anomalies. Today, I generated a messy transaction dataset (`dirty_store_transactions.csv`) and built a step-by-step cleaning pipeline in Pandas.

## Project Files
*   [day9_cleaning.ipynb](day9_cleaning.ipynb): The Jupyter notebook showing the full cleaning process.
*   [dirty_store_transactions.csv](dirty_store_transactions.csv): The raw, messy store data.
*   [cleaned_store_transactions.csv](cleaned_store_transactions.csv): The final, clean dataset ready for analysis.
*   [generate_dirty_data.py](generate_dirty_data.py): The Python script used to inject anomalies into the data.
*   [setup_day9.py](setup_day9.py): The script that compiles and executes the notebook.
*   [linkedin_post.md](linkedin_post.md): Reflection draft for social media.

---

## Dataset Comparison: Before vs. After

| Metric | Before Cleaning (`dirty_store_transactions.csv`) | After Cleaning (`cleaned_store_transactions.csv`) |
| :--- | :--- | :--- |
| **Row Count** | 1,025 | 954 |
| **Column Count** | 10 | 10 |
| **Duplicate Rows** | 25 (15 exact, 10 logical) | 0 |
| **Missing Dates** | 46 (NaN or invalid format like `2018/13/45`) | 0 |
| **Sales Format** | Mixed strings (e.g. `"$1,200.50"`, `"120 USD"`) & negatives | Float64 (cleaned & standardized) |
| **Missing Sales** | 49 nulls + 33 negative anomalies | 0 (Imputed with Category median) |
| **Quantity Format** | Mixed float/int with missing values | Int32 (missing imputed with median `5.0`) |
| **Postal Codes** | Mixed types/strings of different lengths | Standardized 5-character strings (e.g. `'02108'`) |

---

## Data Cleaning Decisions

### 1. Duplicate Removal
*   **Exact duplicates:** Identified and removed 15 rows where all columns were identical (simulating double logging).
*   **Logical duplicates:** Found 10 rows with identical transaction details but assigned a different `Row ID` (simulating data entry retries). Kept the first instance and dropped the rest.

### 2. Fixing Date Formats
*   Dates were recorded in mixed formats (e.g., `YYYY-MM-DD`, `MM/DD/YYYY`, and `DD-MMM-YY`).
*   **Action:** Used `pd.to_datetime(..., format='mixed', errors='coerce')` to parse them.
*   **Decision:** Dropped 46 rows where the date was missing or invalid (like `2018/13/45`), since transaction dates are critical and cannot be imputed.

### 3. Cleaning Casing & Text Typos
*   **Segments & Categories:** Cleaned casing inconsistencies (`consumer` $\rightarrow$ `Consumer`, `furniture` $\rightarrow$ `Furniture`) and corrected spelling typos like `Consumer_typo`.
*   **Country:** Standardized mixed country abbreviations (`US`, `USA`, `United States of America`) to `United States`.
*   **Names:** Handled name trailing spaces and casing.

### 4. Cleaning & Imputing Sales
*   Sales were read as text objects due to symbols (`$`, `,`, `"USD"` text).
*   **Action:** Used string replacement regex to strip non-numeric symbols and converted the column to float.
*   **Decision:** Negative sales (like `-999.0`) were treated as system errors and converted to `NaN`. Imputed all missing and negative sales values using the **median sales value of the respective product category** to preserve the distribution.

### 5. Cleaning & Imputing Quantities
*   Some quantities were loaded as floats (e.g. `3.0`) or were missing.
*   **Action:** Imputed missing values with the median quantity (`5`) and cast the column to `int`.

### 6. Standardizing Postal Codes
*   Some postal codes were read as floats (e.g., `90036.0`) or were short strings missing leading zeros (e.g. `2108` instead of `02108`).
*   **Action:** Formatted all postal codes to 5-digit strings by removing decimal points and padding them with leading zeros. Set missing zip codes to `'Unknown'`.

---

## How to Run
To regenerate the messy dataset and execute the data cleaning pipeline:
```bash
python day9/generate_dirty_data.py
python day9/setup_day9.py
```
This will recreate `dirty_store_transactions.csv`, run the Jupyter notebook `day9_cleaning.ipynb`, and export the cleaned dataset.
