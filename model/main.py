import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Function to load and clean dataset
def get_clean_dataset():
    dataset = pd.read_csv("data/german_credit_data_biased_training [MConverter.eu].csv")

    # Drop unnecessary fields
    dataset.drop(['Sex', 'ForeignWorker', 'Telephone'], axis=1, inplace=True)

    # Define mappings for categorical variables
    category_mappings = {
        "CheckingStatus": {"0_to_200": 0, "less_0": 1, "no_checking": 2, "greater_200": 3},
        "CreditHistory": {"credits_paid_to_date": 0, "prior_payments_delayed": 1, "outstanding_credit": 2,
                          "all_credits_paid_back": 3, "no_credits": 4},
        "LoanPurpose": {"other": 0, "car_new": 1, "furniture": 2, "retraining": 3, 
                        "education": 4, "vacation": 5, "appliances": 6, "car_used": 7, 
                        "repairs": 8, "radio_tv": 9, "business": 10},
        "ExistingSavings": {"100_to_500": 0, "less_100": 1, "500_to_1000": 2, "unknown": 3, "greater_1000": 4},
        "EmploymentDuration": {"less_1": 0, "1_to_4": 1, "greater_7": 2, "4_to_7": 3, "unemployed": 4},
        "OthersOnLoan": {"none": 0, "co-applicant": 1, "guarantor": 2},
        "OwnsProperty": {"savings_insurance": 0, "real_estate": 1, "unknown": 2, "car_other": 3},
        "InstallmentPlans": {"none": 0, "stores": 1, "bank": 2},
        "Housing": {"own": 0, "free": 1, "rent": 2},
        "Job": {"skilled": 0, "management_self-employed": 1, "unskilled": 2, "unemployed": 3},
        "Risk": {"No Risk": 0, "Risk": 1}
    }

    # Apply categorical mappings
    for col, mapping in category_mappings.items():
        dataset[col] = dataset[col].replace(mapping)

    # Convert entire dataset to float
    dataset = dataset.astype(float)

    return dataset

# Function to create and train the machine learning model
def create_model(dataset):
    # Define predictors (independent variables) and target variable
    selected_features = [
        'CheckingStatus', 'CreditHistory', 'LoanPurpose', 
        'ExistingSavings', 'EmploymentDuration', 'OthersOnLoan',
        'OwnsProperty', 'InstallmentPlans', 'Housing', 'Job'
    ]

    X = dataset[selected_features]  # Keep only the selected features
    y = dataset['Risk']  # Target variable

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure 'model' directory exists
    os.makedirs('model', exist_ok=True)

    # Save only the selected feature names
    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(selected_features, f)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Model Accuracy:', accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model

# Main function to execute dataset preprocessing and model training
def main():
    dataset = get_clean_dataset()  # Load and clean dataset
    model = create_model(dataset)  # Train model

    # Save the trained model
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)  # Save as binary file

# Ensures script runs only when executed directly
if __name__ == '__main__':
    main()
