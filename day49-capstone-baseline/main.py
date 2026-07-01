import pandas as pd
import json
from data_generator import generate_customer_data
from preprocessing import DataPreprocessor
from models import ChurnPredictor, CustomerSegmenter

def main():
    print("Starting Capstone Baseline Pipeline...")
    
    # 1. Generate/Load Data
    print("Generating synthetic data...")
    df = generate_customer_data(n_samples=2000)
    df.to_csv("customer_data.csv", index=False)
    
    # 2. Preprocessing
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, processed_df = preprocessor.prepare_data(df)
    
    # 3. Train Churn Predictor
    print("Training Churn Prediction Model...")
    churn_model = ChurnPredictor()
    churn_model.train(X_train, y_train)
    
    accuracy, report = churn_model.evaluate(X_test, y_test)
    print(f"Churn Model Accuracy: {accuracy:.4f}")
    
    churn_model.save_model("churn_model.pkl")
    
    # 4. Train Customer Segmentation
    print("Training Customer Segmentation Model...")
    X_seg_scaled, seg_df = preprocessor.scale_features_for_segmentation(df)
    
    segmenter = CustomerSegmenter(n_clusters=4)
    clusters = segmenter.fit_predict(X_seg_scaled)
    
    seg_df['segment'] = clusters
    seg_df.to_csv("segmented_customers.csv", index=False)
    segmenter.save_model("segmentation_model.pkl")
    
    # 5. Generate Outputs and Metadata
    print("Generating output reports...")
    output_metadata = {
        "churn_model_accuracy": accuracy,
        "classification_report": report,
        "segmentation_clusters_distribution": seg_df['segment'].value_counts().to_dict()
    }
    
    with open("pipeline_metrics.json", "w") as f:
        json.dump(output_metadata, f, indent=4)
        
    print("Baseline prototype pipeline completed successfully.")

if __name__ == "__main__":
    main()
