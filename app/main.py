import streamlit as st # streamlit library for the streamlit framework
import pickle # importing the ml model to prevent it from being trained inside the application otherwise it will bring about perfomamnce issues
import pandas as pd # for dataset loading and preprocessing
import plotly.graph_objects as go # plot visualization (radar chart)
import numpy as np # convering the input_data to an array


# Define dropdown options for user input
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
    "Job": ['skilled', 'management_self-employed', 'unskilled', 'unemployed'],
    
}

  # input dictionary key function used to store the input measurements so as to create the chart and the prediction 
input_dict = {}

# loop each label for their values
# the key select the column in the data associated to an independent variable
# the two values in the above slider_lables list consists of a label and a key consecutively

for label, key in options: 
    input_dict[key] = st.sidebar.slider(
      label, # first value in slider_lables
      min_value=float(0), # minimum value of a label
      max_value=float( dataset[key].max()), # maximum value of a label
      value=float( dataset[key].mean())
    )
    
    return input_dict

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