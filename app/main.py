import streamlit as st # streamlit library for the streamlit framework
import pickle # importing the ml model to prevent it from being trained inside the application otherwise it will bring about perfomamnce issues
import pandas as pd # for dataset loading and preprocessing
import plotly.graph_objects as go # plot visualization (radar chart)
import numpy as np # convering the input_data to an array


# function to fetch the cleaned data of the model (we need clean data for the sidebar component which has the independent variables)
def get_clean_dataset(): # loading and cleaning the data

   dataset = pd.read_csv("data/german_credit_data_biased_training [MConverter.eu].csv") #reading the dataset
    
   # Drop 'Sex', 'ForeignWorker', and 'Telephone' fields if they exist
   dataset.drop(['Sex', 'ForeignWorker', 'Telephone'], axis=1, inplace=True)

   # Define mappings for categorical variables
   CheckingStatus = {
    "0_to_200": 0, "less_0": 1, "no_checking": 2, "greater_200": 3
}

   CreditHistory = {
    "credits_paid_to_date": 0, "prior_payments_delayed": 1, "outstanding_credit": 2,
    "all_credits_paid_back": 3, "no_credits": 4
}

   LoanPurpose = {
    "other": 0, "car_new": 1, "furniture": 2, "retraining": 3, 
    "education": 4, "vacation": 5, "appliances": 6, "car_used": 7, 
    "repairs": 8, "radio_tv": 9, "business": 10
}

   ExistingSavings = {
    "100_to_500": 0, "less_100": 1, "500_to_1000": 2, 
    "unknown": 3, "greater_1000": 4
}

   EmploymentDuration = {
    "less_1": 0, "1_to_4": 1, "greater_7": 2, 
    "4_to_7": 3, "unemployed": 4
}

   OthersOnLoan = {
    "none": 0, "co-applicant": 1, "guarantor": 2
}

   OwnsProperty = {
    "savings_insurance": 0, "real_estate": 1, "unknown": 2, "car_other": 3
}

   InstallmentPlans = {
    "none": 0, "stores": 1, "bank": 2
}

   Housing = {
    "own": 0, "free": 1, "rent": 2
}

   Job = {
    "skilled": 0, "management_self-employed": 1, 
    "unskilled": 2, "unemployed": 3
}

   Risk = {
    "No Risk": 0, "Risk": 1
}

   # Apply transformations to the dataset
   dataset['CheckingStatus'] = dataset['CheckingStatus'].replace(CheckingStatus)
   dataset['CreditHistory'] = dataset['CreditHistory'].replace(CreditHistory)
   dataset['LoanPurpose'] = dataset['LoanPurpose'].replace(LoanPurpose)
   dataset['ExistingSavings'] = dataset['ExistingSavings'].replace(ExistingSavings)
   dataset['EmploymentDuration'] = dataset['EmploymentDuration'].replace(EmploymentDuration)
   dataset['OthersOnLoan'] = dataset['OthersOnLoan'].replace(OthersOnLoan)
   dataset['OwnsProperty'] = dataset['OwnsProperty'].replace(OwnsProperty)
   dataset['InstallmentPlans'] = dataset['InstallmentPlans'].replace(InstallmentPlans)
   dataset['Housing'] = dataset['Housing'].replace(Housing)
   dataset['Job'] = dataset['Job'].replace(Job)
   dataset['Risk'] = dataset ['Risk'].replace(Risk)

   return dataset

# Define dropdown options for user input
def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  
dataset = get_clean_dataset()

 # sliding function for the independent variables (columns) each with a label
options = {
    "CheckingStatus": ['0_to_200', 'less_0', 'no_checking', 'greater_200'],
    "CreditHistory": ['credits_paid_to_date', 'prior_payments_delayed', 'outstanding_credit',
                      'all_credits_paid_back', 'no_credits'],
    "LoanPurpose": ['other', 'car_new', 'furniture', 'retraining', 'education', 'vacation',
                    'appliances', 'car_used', 'repairs', 'radio_tv', 'business'],
    "ExistingSavings": ['100_to_500', 'less_100', '500_to_1000', 'unknown', 'greater_1000'],
    "EmploymentDuration": ['less_1', '1_to_4', 'greater_7', '4_to_7', 'unemployed'],
    "Sex": ['female', 'male'],
    "OthersOnLoan": ['none', 'co-applicant', 'guarantor'],
    "OwnsProperty": ['savings_insurance', 'real_estate', 'unknown', 'car_other'],
    "InstallmentPlans": ['none', 'stores', 'bank'],
    "Housing": ['own', 'free', 'rent'],
    "Job": ['skilled', 'management_self-employed', 'unskilled', 'unemployed'],
    "Telephone": ['none', 'yes'],
    "ForeignWorker": ['yes', 'no']
}

# Function to collect user inputs via Streamlit sidebar
def get_user_inputs():
    st.sidebar.header("Loan Applicant Information")
    input_data = {}

    for feature, choices in options.items():
        input_data[feature] = st.sidebar.selectbox(f"{feature}:", choices)

    return input_data

# Function to load trained model and make predictions
def predict_loan_risk(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))  # Load trained ML model
    encoders = pickle.load(open("model/encoders.pkl", "rb"))  # Load categorical encoders

    # Encode categorical features using stored encoders
    encoded_data = np.array([encoders[col].transform([input_data[col]])[0] for col in input_data]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(encoded_data)
    probability = model.predict_proba(encoded_data)

    return prediction[0], probability

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Loan Risk Predictor",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏦 Loan Risk Predictor Application")
    st.write("Predicts loan approval risk based on financial and personal information.")

    # Get user inputs
    user_inputs = get_user_inputs()

    # Predict loan risk
    if st.sidebar.button("Predict Loan Risk"):
        prediction, probability = predict_loan_risk(user_inputs)

        st.subheader("Loan Risk Prediction")
        if prediction == 0:
            st.write("✅ **No Risk** - The applicant is likely to repay the loan.")
        else:
            st.write("⚠️ **Risk** - The applicant is at high risk of defaulting.")

        st.write(f"**Probability of No Risk:** {probability[0][0]:.2f}")
        st.write(f"**Probability of Risk:** {probability[0][1]:.2f}")

if __name__ == "__main__":
    main()



  


 
if __name__ == '__main__':
  main()