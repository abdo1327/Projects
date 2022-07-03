# cervical cancer classifier 
## Problem
Cervical cancer is one of the dangerous and deadly cancer, which is considered the fourth frequent with more than 500,000 case in 2018 according to WHO. the early diagnosing for the cancer along with the access to the correct and efficient treatment have a significant positive impact on the patient. This project aims to develop a high accuracy classifier which will be using a dataset for more than 850 different case positively and negatively diagnosed with the cervical cancer as helping tool to predict the cancer in early stage. 
The stages to create the classifier will start by obtaining the dataset form [this link]( https://datahub.io/machine-learning/cervical-cancer)    , followed by processing the data and cleaning it by replacing the Nan values and the unnecessary attributes  , visualizing  the  data with the correct visual representation , stating a null hypothesis . 
In stage two the null hypothesis will be tested, the classifier will be trained and tested against the dataset. applying different classification algorithms to find the most suitable accuracy. Investigating the usually classification problems such as overfitting, imbalanced data, biased data

## Data 
The dataset is available for public use from the mentioned link,according to UCI“The dataset was collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela.” .the dataset description is as the following, the data contains 858 samples and 32 attributes as well as four class (Hinselmann, Schiller, Cytology and Biopsy). The same data set had been used by (Choudhury & Won, 2018) by using the Biopsy class as the targeted  value  the aim is to use the other classes .The missing data almost 13.2% from the total data. the data’s features and type showed in table one (Choudhury & Won, 2018) 
![image](https://user-images.githubusercontent.com/91053938/177031169-f3aaf70f-d1db-4f2e-a2f6-f149c9172e74.png)
![image](https://user-images.githubusercontent.com/91053938/177031164-3a95e528-5861-43a3-94a0-2d4097482731.png)

The clean process was done by using python programing language and Pandas library. starting by  replace the massing value ‘?’ from the csv original file with a unique number ‘8529’ to make it easier to deal with it since ‘?’ is string  and need to be converted than replaced so it was easier to replaced in the original file to integer ,therefore no need to converted in python . 
The second step was dropping the unnecessary attributes  which have to types attributes was a lot of missing values (STDs: Time since first diagnosis, STDs: Time since last diagnosis) with more than 90% missing values and the second type is the attoliters with one values ( STDs:cervical condylotomies, STDs:AIDS) . the last step for the clearing process was to replace the missing values with the median in the numerical attributes, and the mode in the categorical attributes.
The targets classes Hinselmann, Schiller, Cytology and Biopsy have positive result test 35,74,44,55 respectively which shows imbalanced data.  The age data shows right skewed distribution which mean that the majority of the subject are young (<40) as it can be shown in (figure 1). The Hormonal Contraceptives is correlated with the target values positively, therefore we can set it as null hypothesis that Hormonal Contraceptives effect the number of cervical cancer case and test in stage 2.
On there hand the smoker dose not showing significant positive result in the targeted classes as well as STD, and IUD.
## workflow 
1.	Hypothesis testing for null hypothesis
2.	Finding the attributes that effect the target values the most (feature selection)
3.	creating the module which include chosen the classifiers algorism, model evolution 
4.	comparing the accuracy of different algorisms such as k-nearest neighbors (KNN), Gaussian Naive Bayes (GNB), Logistic Regression (LR) since it is binary classifier , Decision Tree (DT)

## References 

1. Choudhury, A., & Won, D. (2018). Classification of Cervical Cancer Dataset Abstract.
2. [hyperparameters for classification ML (uesd in the code )](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/) 
3. scikit-learn documentation 
4. [WHO-cervical cancer](https://www.example.com](https://www.who.int/cancer/prevention/diagnosis-screening/cervical-cancer/en/ )

