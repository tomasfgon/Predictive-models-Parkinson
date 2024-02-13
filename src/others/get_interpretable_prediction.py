import os
from fnmatch import fnmatch

import joblib
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
#########FEATURE EXTRACTION

flag_extract_mfcc = False


#directory of audio files for prediction
predict_dir = "data/Italian_PD"

#path of trained model to load
models_path = "data/models/italian/interpretable/"



def getListOfAudioPaths(dir_path):
    full_audiofile_paths= []
    pattern = "*.wav"

    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if fnmatch(name, pattern):
                full_audiofile_paths.append(os.path.join(dir_path,name))
    return(full_audiofile_paths)



def getFeatures(voiceID, f0min, f0max, unit):
    try:
        sound = parselmouth.Sound(voiceID) # read the sound
        duration = call(sound, "Get total duration") # duration
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object


        meanF0 = call(pitch, "Get mean", 0, 0, unit) #MDVP:Fo(Hz)
        # get standard deviation
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) 
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0) #HNR
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) #MDVP:Jitter(%)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) #MDVP:Jitter(Abs)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) #MDVP:RAP
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) #MDVP:PPQ
        ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) #Jitter:DDP
        localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #MDVP:Shimmer
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #MDVP:Shimmer(dB)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #Shimmer:APQ3
        aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) #Shimmer:APQ5
        apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        
        #print("file " + voiceID + " was read successfuly")
        return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
    except Exception as exc:
        print(exc)
        print("FEATURES - file " + voiceID + " was not read because it is not a sound file - ")
        return -1

def extract_mfcc(voiceID):
        try:
        
            """
            Extracts the mel frequency ceptral coefficients from the voice sample

            Parameters:
            voiceID : .wav file
                the voice sample we want to extract the features from
            """

            sound = parselmouth.Sound(voiceID)
            mfcc_object = sound.to_mfcc(number_of_coefficients=12) #the optimal number of coeefficient used is 12
            mfcc = mfcc_object.to_array()
            mfcc_mean = np.mean(mfcc.T,axis=0)
            [mfcc0, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12] = mfcc_mean
            return mfcc0, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12
        except Exception as exc:
            print(exc)
            print("MFCC - file " + voiceID + " was not read because it is not a sound file - ")
            return -1
        

#get features for testfile
def getFeaturesFromFile(path):
    #extract features from audiofile
    testfile_outname = (os.path.basename(path) + '_original_features.csv')


    print("Reading File - " + path)
    values = getFeatures(path, 75, 500, "Hertz")

    mfcc_values = -1
    
    if flag_extract_mfcc:
        mfcc_values = extract_mfcc(path)
        original_features_df = pd.DataFrame(columns=['Duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12'])

    else:
        original_features_df = pd.DataFrame(columns=['Duration', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer'])


    if (values!=-1):
        duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer = values
        
        if flag_extract_mfcc:
            mfcc0,mfcc1,mfcc2,mfcc3,mfcc4,mfcc5,mfcc6,mfcc7,mfcc8,mfcc9,mfcc10,mfcc11,mfcc12 = mfcc_values

        if (flag_extract_mfcc and mfcc_values !=-1):
            original_features_df.loc[len(original_features_df)] = [duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, mfcc0, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12]
        else:
            original_features_df.loc[len(original_features_df)] = [duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]


    
    # if not os.path.exists(testfile_table_path):
    #     os.makedirs(testfile_table_path, exist_ok=False)

    # original_features_df.to_csv(os.path.join(testfile_table_path, testfile_outname),index=False)
    return original_features_df



#PROCESS

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

def dropnas(features_df):
    features_df.dropna(subset=['meanF0'], inplace=True)
    features_df.dropna(subset=['stdevF0'], inplace=True)
    features_df.dropna(subset=['hnr'], inplace=True)
    features_df.dropna(subset=['localJitter'], inplace=True)
    features_df.dropna(subset=['localabsoluteJitter'], inplace=True)
    features_df.dropna(subset=['rapJitter'], inplace=True)
    features_df.dropna(subset=['ppq5Jitter'], inplace=True)
    features_df.dropna(subset=['ddpJitter'], inplace=True)
    features_df.dropna(subset=['localShimmer'], inplace=True)
    features_df.dropna(subset=['localdbShimmer'], inplace=True)
    features_df.dropna(subset=['apq3Shimmer'], inplace=True)
    features_df.dropna(subset=['aqpq5Shimmer'], inplace=True)
    features_df.dropna(subset=['apq11Shimmer'], inplace=True)
    features_df.dropna(subset=['ddaShimmer'], inplace=True)

    if flag_extract_mfcc:
        features_df.dropna(subset=['mfcc0'], inplace=True)
        features_df.dropna(subset=['mfcc1'], inplace=True)
        features_df.dropna(subset=['mfcc2'], inplace=True)
        features_df.dropna(subset=['mfcc3'], inplace=True)
        features_df.dropna(subset=['mfcc4'], inplace=True)
        features_df.dropna(subset=['mfcc5'], inplace=True)
        features_df.dropna(subset=['mfcc6'], inplace=True)
        features_df.dropna(subset=['mfcc7'], inplace=True)
        features_df.dropna(subset=['mfcc8'], inplace=True)
        features_df.dropna(subset=['mfcc9'], inplace=True)
        features_df.dropna(subset=['mfcc10'], inplace=True)
        features_df.dropna(subset=['mfcc11'], inplace=True)
        features_df.dropna(subset=['mfcc12'], inplace=True)
    return features_df


#APPLY MODEL


def loadModelAndPredict(reduced_features_df):
    #loads saved sccaler and features from trained model
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
    voting = {"name":"Voting", "model":joblib.load(os.path.join(models_path , 'voting.joblib'))}



    #Scaling
    # Load the saved MinMaxScaler
    scaler = joblib.load(scaler_path)

    # Load the column names
    features_names_path = features_names_path
    features_names = joblib.load(features_names_path)


    # Transform the new data using the loaded scaler
    new_data_scaled_df = pd.DataFrame(scaler.transform(reduced_features_df.values), columns=features_names, index=reduced_features_df.index)


    models = [lr, knn, nb, svm, rf, bgcl, adabc, xgbc, voting]
    
    predictions = pd.DataFrame(columns=['Model', 'Predicted_Probability', 'Predicted_Class'])

 
    # Use the trained logistic regression model for predictions
    for i, model_dict in enumerate(models):
        y_predict_proba = model_dict["model"].predict_proba(new_data_scaled_df)
        positive_class_prob = y_predict_proba[:, 1]
    
        predictions.loc[i] = {'Model': model_dict["name"], 'Predicted_Probability': positive_class_prob, 'Predicted_Class': model_dict["model"].predict(new_data_scaled_df)}

    print(predictions)
    return predictions.loc[len(predictions)-1]
    

def process_data(original_features_df):
    features_df = original_features_df
    features_df = dropnas(features_df)

    features_df.drop('Duration', axis=1, inplace=True)

    removed_outliers = outliearTreatment(features_df)

    reduced_features_df = removed_outliers.copy()

    reduced_features_path = (os.path.basename(path) + '_reduced_features.csv')

    

    return reduced_features_df, reduced_features_path

#Principal function
def getProbabilities(file_paths):

    #EXTRACTION

    #Extract features from given file
    original_features_df = getFeaturesFromFile(file_paths)
    
    #PROCESSING
    reduced_features_df, reduced_features_path = process_data(original_features_df)
    

    #save tables
    #reduced_features_df.to_csv(os.path.join(testfile_table_path, reduced_features_path),index=False)


    #APPLY MODEL
    return loadModelAndPredict(reduced_features_df)



getListOfAudioPaths(predict_dir)
getProbabilities(getListOfAudioPaths(predict_dir))