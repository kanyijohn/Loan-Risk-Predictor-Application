import streamlit as st # streamlit library for the streamlit framework
import pickle # importing the ml model to prevent it from being trained inside the application otherwise it will bring about perfomamnce issues
import pandas as pd # for dataset loading and preprocessing
import plotly.graph_objects as go # plot visualizatio (radar chart)
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



# sidebar component for input data(cell measurements)
def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  
  data = get_clean_dataset()

  # sliding function for the independent variables (columns) each with a label 
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  # input dictionary key function used to store the input measurements so as to create the chart and the prediction 
  input_dict = {}

# loop each label for their values
# the key select the column in the data associated to an independent variable
# the two values in the above slider_lables list consists of a label and a key consecutively

  for label, key in slider_labels: 
    input_dict[key] = st.sidebar.slider(
      label, # first value in slider_lables
      min_value=float(0), # minimum value of a label
      max_value=float(data[key].max()), # maximum value of a label
      value=float(data[key].mean())
    )
    
  return input_dict


  

# function used to get the values(cell measurements) from the dictionary of values- for plot visualization
def get_radar_chart(input_data):
  
  input_data = (input_data) # calls the function used for scaling the data values making the radar chart more usable
  
  # represents the independent variables of the dataset for the 10 values
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()
  
  # Mean Value Trace
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories, # angular values(categories- the independent variables) listed below
        fill='toself', # colour for the trace
        name='Mean Value' # name of the trace (key of the radar chart)
  ))

  # Standard Error Trace
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))

  # Worst Value Trace
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1] # scaling the data to stadardize the input features (independent variable values) hence make the radar chart visually recognizable and able to analyse
      )),
    showlegend=True
  )
  
  return fig

# function for the prediction column
def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb")) # importing the model from (using) pickle [rb- read binary model]
  scaler = pickle.load(open("model/scaler.pkl", "rb")) # importing the scaler from (using) pickle [rb- read binary model]
  
  # converting all values in the input dictionary (input _data) containing the key and their corresponding values into an array to make the prediction when updating the inputs
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
 
  
  prediction = model.predict(input_array_scaled) # scaler logic where 0 indicates benign and 1 indicates malignant
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  # diagnosis prediction
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0]) # benign prediction probability 
  st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1]) # malignant prediction probability 
  
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
    radar_chart = get_radar_chart(input_data) # arguments taking the dictionary of values from the sidebar (data input-cell measurements)
    st.plotly_chart(radar_chart) # passing in the figure element- the figure function

  # column 2- cancer prediction display  
  with col2:
    add_predictions(input_data)


 
if __name__ == '__main__':
  main()