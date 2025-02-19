import streamlit as st # streamlit library for the streamlit framework
import pickle # importing the ml model to prevent it from being trained inside the application otherwise it will bring about perfomamnce issues
import pandas as pd # for dataset loading and preprocessing
import plotly.graph_objects as go # plot visualization (radar chart)
import numpy as np # convering the input_data to an array


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

   dataset = dataset.astype(float)

    
   return dataset


# Sidebar component for user inputs
def add_sidebar():
    st.sidebar.header("Loan Applicant Information")  # Sidebar Title
    
    dataset = get_clean_dataset()  # Load cleaned dataset
    
    # Define options for categorical variables
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
    
    input_dict = {}  # Dictionary to store user inputs

    # Loop through each feature and create a dropdown selectbox
    for label, choices in options.items():
        input_dict[label] = st.sidebar.selectbox(f"{label}:", choices)  # ✅ Corrected f-string

    return input_dict  # ✅ Now correctly inside the function

import plotly.graph_objects as go

import plotly.graph_objects as go

# Function to generate a pie chart
def get_pie_chart(input_data):
    # Define categories for visualization
    categories = ['Checking Status', 'Credit History', 'Loan Purpose', 
                  'Existing Savings', 'Employment Duration', 'Others on Loan',
                  'Owns Property', 'Installment Plans', 'Housing', 'Job']
    
    # Get corresponding values from input data
    values = [
        input_data['CheckingStatus'], input_data['CreditHistory'], input_data['LoanPurpose'],
        input_data['ExistingSavings'], input_data['EmploymentDuration'], input_data['OthersOnLoan'],
        input_data['OwnsProperty'], input_data['InstallmentPlans'], input_data['Housing'],
        input_data['Job']
    ]

    # Create a pie chart
    fig = go.Figure(data=[go.Pie(labels=categories, values=values, hole=0.3)])

    # Update layout
    fig.update_layout(title_text="Loan Applicant's Financial Profile Distribution")

    return fig




# function for the prediction column
def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb")) # importing the model from (using) pickle [rb- read binary model]
 
  
  # converting all values in the input dictionary (input _data) containing the key and their corresponding values into an array to make the prediction when updating the inputs
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  
  prediction = model.predict(input_array) # scaler logic where 0 indicates benign and 1 indicates malignant
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  # Loan Risk prediction
  if prediction[0] == 0:
    st.write("<span class='Risk benign'>No Risk</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='Risk malicious'>Risk</span>", unsafe_allow_html=True)
    
  
  st.write(f"**Probability of No Risk:** {model.predict_proba(input_array)[0][0]}")  # ✅ Fixed
  st.write(f"**Probability of Risk:** {model.predict_proba(input_array)[0][0]}")  # ✅ Fixed

  
  st.write("This application is aimed to assist medical professionals in making Breast Cancer diagnosis, NOT as a substitute for a professional diagnosis.")


# page configuration- main function 
def main():
  st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  # importig the style.css file- under page configuration
  with open("assets/style.css") as f: # opening the assets folder containing the style.css as f for file
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True) # makes the style.css be read as a markdown html file
  
  input_data = add_sidebar() # returns the data values from the sidebar where there exists dictionary of values(the key) of the independent variables
  
  with st.container():
    st.title("Breast Cancer Diagnosis Predictor")
    st.write("This application predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can update the cell measurements using the sliders on the sidebar.")
  
  col1, col2 = st.columns([4,1]) # creating the columns(first column(radar chart column) should be 4 times larger than the second column(diagnosis prediction column))
  
  # column 1- plot visualization display
  with col1:
    radar_chart = get_pie_chart(input_data) # arguments taking the dictionary of values from the sidebar (data input-cell measurements)
    st.plotly_chart(radar_chart) # passing in the figure element- the figure function

  # column 2- cancer prediction display  
  with col2:
    add_predictions(input_data)




if __name__ == "__main__":
    main()