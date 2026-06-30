# Day 48: Building the Data Pipeline for Your Customer Intelligence Platform

## Overview
This phase of the capstone project focuses on designing and implementing a robust data preprocessing pipeline. In a production setting, machine learning models require a consistent stream of clean, feature-rich data. This repository contains the components to transform raw customer and transactional data into a unified, model-ready dataset.

## Repository Structure
- `data/raw/`: Raw customer and transactional data generated via `src/data_generation.py`.
- `data/cleaned/`: The preprocessed dataset outputted by the pipeline.
- `src/preprocessing.py`: Contains custom scikit-learn transformers (`OutlierClipper`, `DateFeaturesExtractor`, `FeatureAggregator`) for modular preprocessing.
- `src/data_generation.py`: Script to generate synthetic datasets for the capstone.
- `preprocessing_pipeline.ipynb`: Jupyter notebook demonstrating the step-by-step pipeline execution and visualization of the data integration process.
- `run_pipeline.py`: Python script equivalent of the notebook to execute the pipeline and generate the clean dataset.
- `pipeline_architecture.md`: Detailed documentation on the architecture and design decisions of the preprocessing pipeline.
- `linkedin_post.txt`: A summary of accomplishments for Day 48.

## How to Run
1. Generate the raw data:
   ```bash
   python src/data_generation.py
   ```
2. Execute the preprocessing pipeline:
   ```bash
   python run_pipeline.py
   ```
   *Alternatively, explore the `preprocessing_pipeline.ipynb` notebook to see the steps iteratively.*

## Key Learnings
- Constructing scikit-learn compatible custom transformers for seamless pipeline integration.
- Leveraging `ColumnTransformer` to handle disparate data types (numerical and categorical) concurrently.
- Developing modular architecture to ensure reproducibility in data engineering workflows.
