# Project-7-Singapore-resale-flat-price-prediction

**DATA COLLECTION:**
The project begins with a data Collection such as flat type,flat model,floor sqrt,resale price etc...


**DATA PREPROCESSING**
After collecting data,next steps is data preprocessing, which involves handling missing values, removing duplicate values, and converting data types to prepare the dataset for analysis and modeling.

**ANALYSIS THE DATA**
Exploratory Data Analysis (EDA): 
UNIVARIATE ANALYSIS:
Analysis done by taking one varibale such as flat type,flat model etc for categorical column.
Analysis done by taking on variable such as resale price,minimum storey,maximum storey,floor are sqrt.

BIVARIATE ANALYSIS:
Analysis done using pairplots -floor_area_sqm,resale_price,maximum_storey,minimum_storey,year.

MULTIVARIATE ANALYSIS:
Analysis numerical variable to find a correlation between multiple variable using heatmap.

**MODEL TRAINING**

1**ENCODING**
using ordinal encode for converting categorical variable into numerical variable for model training.

2**__init__ method**
*Initializes the list of models.
*Splits the dataset into training and testing sets.
*train_and_evaluate_model function:

**Trains a given model**
Predicts the target values for both training and testing sets.
Calculates and returns the evaluation metrics.
Catches and prints any exceptions that occur during model training/evaluation.

**all_model method**
Iterates over all the models.
Uses the train_and_evaluate_model function to train and evaluate each model.
Compiles the results into a DataFrame and returns it.


