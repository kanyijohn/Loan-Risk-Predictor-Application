import pandas as pd #pip install pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report #accuracy score and detailed metrics report
import pickle as pickle #used for serializing and deserializing Python objects into a format that can be easily stored or transmitted and deserialization is the reverse process

def create_model(dataset): # creating the model

    X = dataset.drop(['Risk'], axis=1) #predictors (independent variables)-all columns except diagnosis 
    y = dataset['Risk'] #target variable

    dataset = dataset.astype(float)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)


    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
  
    return model




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




# main function( running the main function- the functions contained in the main function are being called for execution under the main function)
def main():
    dataset = get_clean_dataset() 
    
    model = create_model(dataset)
    
    with open('model/model.pkl', 'wb') as f:
     pickle.dump(model, f) # dumps the model object (i.e logical regression model) and writes it to the file f. The model is then saved as a binary .pkl file.
    
    
    






if __name__ == '__main__': #helps code structure such that certain parts (like main()) only run when the script is executed directly, not when it is imported
  main()  