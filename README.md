# Predictive models in the diagnosis of Parkinson's disease through analysis of voice

## Interpretable features

### features.ipynb
 - Allows extraction of traditional interpretable features using Parselmouth
 - ```flag_[dataset]_db = True``` to extract features of [dataset]
 - generates tables of original features

 ### processing.ipynb
 - processes the data and generates new tables with clean data

 ### modeling.ipynb
 - Train and test data from dataset using data from processed tables
 - scales data
 - choose dataset with ```selected_option = MyEnum.[dataset]```
- displays accuracy tables and plots of each trained/tested algorithm - saves model and scaler in ```/data/models/[dataset]/interpretable```



 ## Non-interpretable features

 ### non-interpretable_features.ipynb
 - Allows extraction of embeddding non-interpretable features
 - choose feature extraction method and dataset
     ```
     selected_dataset = Dataset.[dataset]
 
     selected_extractor = Extractor.[extracting method]
     ```
- saves features in numpy array in .pkl file ```/data/non_interpretable_features``` 
- features are saved in the format ```[dataset][extractor number]_[hc/pd].pkl```
  - extractor numbers: 0 -> x-vector; 1 -> trillsson; 2 -> wav2vec; 3-> hubert

### non-interpretable_classifier.ipynb
- trains and tests models for non-interpretable features
- choose feature extraction method and dataset
     ```
     selected_dataset = Dataset.[dataset]
 
     selected_extractor = Extractor.[extracting method]
     ```
- displays accuracy tables and plots of each trained/tested algorithm

 # src/script files
 ### get_prediction.py
 - generates interpretable feature based predictions for all the audio files inside ```predict_dir``` directory by using one of the trained models
 - choose which model to use by setting directory ``` models_path = "data/models/[dataset folder]/interpretable/" ```


 ### test_different_dataset.ipynb
 - tests prediction accuracy of trained model on an unseen dataset
 - 
    ```
    selected_model = Model.[training dataset]

    #table to test predictions
    file_path = "data/tables/[unseen dataset]_reduced_features.csv"
    ```

## data_comparator.ipynb
- provides plots to compare average values of interpretable features for different datasets 
