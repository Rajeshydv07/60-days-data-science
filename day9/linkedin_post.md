Day 9 of #60DaysOfDataScience: Cleaning Messy Real-World Data 🧹

Everyone talks about training machine learning models, but the reality is that 80% of data science is data cleaning. Today I focused on handling messy, incomplete, and noisy transactional data.

I generated a custom retail dataset containing realistic errors to build a clean pipeline using Pandas:

1️⃣ Duplicate Records: Removed both exact duplicates and logical duplicates (matching transaction details logged under different Row IDs).
2️⃣ Mixed Date Formats: Handled mixed string formats (YYYY-MM-DD, MM/DD/YYYY, and DD-MMM-YY) using pandas `format='mixed'`, and safely dropped rows with invalid dates.
3️⃣ Casing & Spelling Typos: Fixed inconsistent categories (like 'Consumer' vs 'consumer') and resolved typos.
4️⃣ Currency & Numeric Conversions: Stripped currency symbols and text (e.g. '$1,200.50 USD') using regex and imputed missing/negative sales using product category medians.
5️⃣ Postal Codes: Standardized zip codes to 5-digit strings, ensuring lost leading zeros (e.g. for Boston zip codes) were padded back.

This process highlights how critical data formatting is. If your text casing is off or dates are misaligned, your downstream models or aggregations will yield incorrect conclusions.

Clean data is the foundation of reliable decisions! 🚀

You can check out my cleaning notebook and the raw/cleaned datasets here:
https://github.com/Rajeshydv07/ABtalksDS/tree/main/day9

#DataScience #MachineLearning #Pandas #Python #DataCleaning #Analytics #LearningInPublic
