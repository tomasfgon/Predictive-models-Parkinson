from enum import Enum, auto
import os
from fnmatch import fnmatch

import joblib
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from sklearn.metrics import classification_report

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)


class Model(Enum):
    ITALIAN = auto()
    MDVR = auto()
    AH = auto()
    CZECH = auto()


class Features(Enum):
    INTERPRETABLE = auto()
    NON_INTERPRETABLE = auto()



#use model based on which dataset
selected_model = Model.CZECH


#which feature table to test predictions
file_path = "data/tables/italian_reduced_features.csv"




if selected_model == Model.ITALIAN:
    dataset = "italian"
if selected_model == Model.MDVR:
    dataset = "mdvr"
if selected_model == Model.AH:
    dataset = "ah"
if selected_model == Model.CZECH:
    dataset = "czech"



models_path = "data/models/"+dataset+"/interpretable"




def outliearTreatment(df):
    '''
    Any values greater than the whisker (3IQ) are set to the whisker value, 
    and any values lower than the LowerBound (1IQ) are set to the LowerBound.
    '''
    cols = list(df.columns)
    for columnName in cols:
        Q1 = df[columnName].quantile(0.25)
        Q3 = df[columnName].quantile(0.75)
        IQR = Q3 - Q1
        whisker = Q1 + 1.5 * IQR
        LowerBound = Q1- 1.5 * IQR
        df.loc[:, columnName] = df[columnName].apply(lambda x: whisker if x > whisker else x)
        df.loc[:, columnName] = df[columnName].apply(lambda x : LowerBound if x<LowerBound else x)
    return df


def ModelsTest(reduced_features_df, y_test):
    scaler_path = os.path.join(models_path , 'scaler.save') 
    features_names_path = os.path.join(models_path , 'features_names.joblib') 


    # Load the trained logistic regression model 



    lr = {"name":"Logistic Regression", "model":joblib.load(os.path.join(models_path , 'lr_2.joblib'))}
    knn = {"name":"KNN", "model":joblib.load(os.path.join(models_path , 'knn_2.joblib'))}
    nb = {"name":"Naive Bayes", "model":joblib.load(os.path.join(models_path , 'nb.joblib'))}
    svm = {"name":"SVM", "model":joblib.load(os.path.join(models_path , 'svm_2.joblib'))}
    rf = {"name":"Random Forest", "model":joblib.load(os.path.join(models_path , 'rf_2.joblib'))}
    bgcl = {"name":"Bagging", "model":joblib.load(os.path.join(models_path , 'bgcl_2.joblib'))}
    adabc = {"name":"AdaBoost", "model":joblib.load(os.path.join(models_path , 'adabc_2.joblib'))}
    xgbc = {"name":"XGBoost", "model":joblib.load(os.path.join(models_path , 'xgbc_2.joblib'))}
    nn = {"name":"Neural Network", "model":joblib.load(os.path.join(models_path , 'nn.joblib'))}
    voting = {"name":"Voting", "model":joblib.load(os.path.join(models_path , 'voting.joblib'))}



    #Scaling

    # Load the saved MinMaxScaler
     
    scaler = joblib.load(scaler_path)

    # Load the column names
    features_names_path = features_names_path
    features_names = joblib.load(features_names_path)

    # Assuming you already have a DataFrame named new_data_df
    # Ensure that the new data has the same columns as the original data
    # If not, you might need to preprocess it accordingly

    # Transform the new data using the loaded scaler
    #same scaling done in modeling
    new_data_scaled_df = pd.DataFrame(scaler.transform(reduced_features_df.values), columns=features_names, index=reduced_features_df.index) 


    models = [lr, knn, nb, svm, rf, bgcl, adabc, xgbc, nn, voting]

    for model in models:

        y_pred = model["model"].predict(new_data_scaled_df)
        print(model["name"])
        print(classification_report(y_test, y_pred))

        #pd.set_option('display.max_rows', None)
        #table = predictedVsActualTable(model["model"], new_data_scaled_df, y_test)
        #pva_plot_lr_2 = predictedVsActualPlot(table)
        #print(table)





# Define the columns to keep in X_test
features_to_keep = ["meanF0", "stdevF0", "hnr", "localJitter", "localabsoluteJitter",
                    "rapJitter", "ppq5Jitter", "ddpJitter", "localShimmer", "localdbShimmer",
                    "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer", "ddaShimmer"]

# Define the columns to keep in y_test
target_column = ["PD"]

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create X_test DataFrame with selected features
X_test = df[features_to_keep]


# Create y_test DataFrame with the target column
y_test = df[target_column]

# Display the first few rows of X_test and y_test for verification
print("X_test:")
print(X_test)

print("\ny_test:")
print(y_test)

removed_outliers = outliearTreatment(X_test)

ModelsTest(removed_outliers, y_test)