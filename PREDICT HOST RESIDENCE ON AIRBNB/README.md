# Predict host residence on Airbnb  
## Problem Overview and objective
USA’s Capital Gains Tax laws state that property owners who occupy their property are allowed a tax exemption, Some owners misrepresent their property in order to abuse the exemption law.
The project aims to:
1. create a model/s that effectively classifies the property types based on available data on rent listings and reviews.
2. evaluate, document and report the performance of the model/s.
## Project Approach
The workflow for the project can be summarized as the following :
- Create different classifications models.
- Evaluate the models' performance.
- Compare the models' performance for different aspect.
## Project Method and algorithm 
Dataset description : The used dataset was [Sydney Airbnb dataset](https://www.kaggle.com/datasets/tylerx/sydney-airbnb-open-data) which include 6.5M rows each row represent feedback from the guest for a specific property as well as property and visit related features such as date of the visit , property type, properties neighborhoods.
The main issues for this dataset:
- There is no label assigned to it to be used for classifications phase
- The size of the dataset which will be very inefficient and computationally expensive to work with it.

Primary data pre-processing :
- Model training and testing data: 2200 rows contain the guest's review and manually generated labels. (the owner is not living in the property = 0, living in the property =1, unknown=2)
- Prediction dataset: 7500 rows contain the guest's review to be used by the models later to evaluate the running time during the prediction.

Selected models:
- Traditional machine learning (logistic regression, random forest, KNN, SVM)
- RNN

## For Traditional machine learning models workflow
1. Data pre-processing:
 - cleaning the reviews
 - convert to lower case
 - tokenizing and applying stop word
 - stemming
 - Lemmatizing
2. Feature creating applying Tf-Idf.
3. Split to training and testing set.
4. Using the training set to train the model and hype tuning the parameters for each model.
5. Using the test set to evaluate the model.
## For RNN Workflow 
1. Data pre-processing
 - Dealing with words outside the vocabulary
 - Process variable length sequences
 - word embedding

2. Model architecture
 - Embedding layer
 -  RNN
 - Linear Layer

3. Build model
 - Train
 - Evaluate

4. Prediction

![image](https://user-images.githubusercontent.com/91053938/177035769-d4025e01-f176-4b07-a44e-b550d8331738.png)

## The Results 
   Based on performance result the selected models for deployment are  KNN and RNN
![image](https://user-images.githubusercontent.com/91053938/177035863-ec7ccba7-46c3-426f-97e5-727fd080b3d9.png)

# The only code uploaded here is the main code with a brief explanation for the used functions and there poupous which can be found in the documentation file. 
