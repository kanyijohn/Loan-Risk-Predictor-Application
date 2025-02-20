import streamlit as st  # Streamlit framework
import pickle  # Importing the ML model
import pandas as pd  # For dataset handling
import plotly.graph_objects as go  # Visualization
import numpy as np  # Converting input data to an array

# ✅ Set Streamlit Page Configuration at the VERY TOP
st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Function to load and clean dataset
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
        "Job": {"skilled": 0, "management_self-employed": 1, "unskilled": 2, "unemployed": 3}
    }

    # Apply categorical mappings
    for col, mapping in category_mappings.items():
        dataset[col] = dataset[col].replace(mapping)

    # Convert entire dataset to float
    dataset = dataset.astype(float)

    return dataset

# ✅ Sidebar component for user inputs
def add_sidebar():
    st.sidebar.header("Loan Applicant Information")

    options = {
        "CheckingStatus": ['0_to_200', 'less_0', 'no_checking', 'greater_200'],
        "CreditHistory": ['credits_paid_to_date', 'prior_payments_delayed', 'outstanding_credit',
                          'all_credits_paid_back', 'no_credits'],
        "LoanPurpose": ['other', 'car_new', 'furniture', 'retraining', 'education', 'vacation',
                        'appliances', 'car_used', 'repairs', 'radio_tv', 'business'],
        "ExistingSavings": ['100_to_500', 'less_100', '500_to_1000', 'unknown', 'greater_1000'],
        "EmploymentDuration": ['less_1', '1_to_4', 'greater_7', '4_to_7', 'unemployed'],
        "OthersOnLoan": ['none', 'co-applicant', 'guarantor'],
        "OwnsProperty": ['savings_insurance', 'real_estate', 'unknown', 'car_other'],
        "InstallmentPlans": ['none', 'stores', 'bank'],
        "Housing": ['own', 'free', 'rent'],
        "Job": ['skilled', 'management_self-employed', 'unskilled', 'unemployed']
    }

    input_dict = {}
    for label, choices in options.items():
        input_dict[label] = st.sidebar.selectbox(f"{label}:", choices)

    return input_dict

# ✅ Function to generate the Pie Chart
def get_pie_chart(prediction_probabilities):
    labels = ['No Risk', 'Risk']
    values = [prediction_probabilities[0][0], prediction_probabilities[0][1]]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title_text="Loan Risk Prediction Probability")

    return fig

# ✅ Function for Loan Risk Prediction
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))  
    feature_names = pickle.load(open('model/feature_names.pkl', 'rb'))  

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
        "Job": {"skilled": 0, "management_self-employed": 1, "unskilled": 2, "unemployed": 3}
    }

    numerical_input = {col: category_mappings[col][input_data[col]] for col in input_data}
    ordered_input = [numerical_input[feature] for feature in feature_names]
    input_array = np.array(ordered_input).reshape(1, -1)

    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    st.subheader("Loan Risk Prediction")
    st.write("Prediction Result:")

    if prediction[0] == 0:
        st.write("✅ **No Risk** - The applicant is likely to repay the loan.")
    else:
        st.write("⚠️ **Risk** - The applicant is at high risk of defaulting.")

    st.write(f"**Probability of No Risk:** {probability[0][0]:.2f}")
    st.write(f"**Probability of Risk:** {probability[0][1]:.2f}")

    st.plotly_chart(get_pie_chart(probability))

# ✅ Main App Function
def main():
    input_data = add_sidebar()
    st.title("🏦 Loan Risk Predictor Application")
    st.write("Predicts loan approval risk based on financial and personal information.")

    add_predictions(input_data)

if __name__ == "__main__":
    main()
