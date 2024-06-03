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

**step1:**ENCODING****
using ordinal encode for converting categorical variable into numerical variable for model training.

**step2__init__ method**
*Initializes the list of models.Splits the dataset into training and testing sets.train_and_evaluate_model function:

**Trains a given model**
Predicts the target values for both training and testing sets.
Calculates and returns the evaluation metrics.
**all_model method**
Iterates over all the models.
Uses the train_and_evaluate_model function to train and evaluate each model.
Compiles the results into a DataFrame and returns it.

**step3 Data Preparation:**
The input features (x) and the target variable (y) are extracted from a DataFrame, likely in preparation for a machine learning task.
Model Initialization: An instance of a regression model class (possibly custom or from a library) is created, passing the input features and target variable to it.
Model Training and Evaluation: The all_model() method of the regression model instance is called. That method trains and evaluates various regression models using the provided data. The specifics of the models trained and evaluation metrics used are handled within the all_model() method.
Result Storage: The results or trained models generated by the all_model() method are stored in the variable r_models.
*r_models['Train_Mean_Square_Error']: This retrieves the mean squared error values from the training set for each regression model stored in the r_models variable. It accesses the column named 'Train_Mean_Square_Error' from the DataFrame or data structure containing the model evaluation results.
*r_models['Test_Mean_Square_Error']: This retrieves the mean squared error values from the test set for each regression model stored in the r_models variable. It accesses the column named 'Test_Mean_Square_Error' from the DataFrame or data structure containing the model evaluation results.
Model Initialization: An XGBoost regressor model is instantiated using default hyperparameters.

**step4:Model Training**: 
The XGBoost regressor model is trained using the training data (x_train and y_train). During training, the model learns the relationships between the input features and the target variable.
**Prediction Generation:**
Predictions are made on the test data (x_test) using the trained model, generating y_test_pred.
Predictions are also made on the training data (x_train), generating y_train_pred. This step is usually for diagnostic purposes.
Performance Evaluation:
Mean squared error (MSE) is calculated for both the training and test predictions.
MSE quantifies the average squared difference between the actual and predicted values, providing a measure of model performance.
Output:
**Parameter Grid Definition:**
**param_grid**: A dictionary defining the grid of hyperparameters to search over. It specifies different values to try for learning_rate, max_depth, and n_estimators.
Model Initialization:
**xgb_reg** = XGBRegressor(): Creates an instance of the XGBoost regressor with default hyperparameters. This is the model that will be tuned using hyperparameter optimization.
Grid Search:
**grid_search** = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3): Initializes a grid search cross-validation object (GridSearchCV). It takes the XGBoost regressor (xgb_reg), the parameter grid (param_grid), the scoring metric (negative mean squared error in this case, specified by 'neg_mean_squared_error'), and the number of cross-validation folds (3-fold cross-validation, specified by cv=3).
grid_result = grid_search.fit(x_train, y_train): Performs the grid search on the training data (x_train and y_train). It searches for the best combination of hyperparameters based on the specified scoring metric.
Best Model Selection:
**best_params** = grid_result.best_params_: Retrieves the best set of hyperparameters found during the grid search.
best_model = grid_result.best_estimator_: Retrieves the best model (XGBoost regressor) based on the best hyperparameters.
Model Evaluation:
**y_pred** = best_model.predict(x_test): Uses the best model to make predictions on the test set (x_test).
mse = mean_squared_error(y_test, y_pred): Calculates the mean squared error (MSE) between the actual target values (y_test) and the predicted values (y_pred) on the test set.




*XGBoost regressor model (XGBRegressor) with hyperparameters specified and evaluates its performance on the test set. Here's a breakdown of what each part does:

Model Initialization and Training:

xgb_model = XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=200): Creates an instance of the XGBoost regressor with specified hyperparameters (learning_rate=0.1, max_depth=5, n_estimators=200).
xgb_model.fit(x_train, y_train): Trains the XGBoost regressor model on the training data (x_train and y_train).
Prediction Generation:

y_pred = xgb_model.predict(x_test): Uses the trained model to make predictions on the test data (x_test). The predicted values are stored in the variable y_pred.
Model Evaluation:

mse = mean_squared_error(y_test, y_pred): Calculates the mean squared error (MSE) between the actual target values (y_test) and the predicted values (y_pred) on the test set.
rmse = np.sqrt(mse): Calculates the root mean squared error (RMSE) from the MSE.
mae = mean_absolute_error(y_test, y_pred): Calculates the mean absolute error (MAE) between the actual and predicted values on the test set.
r2 = r2_score(y_test, y_pred): Calculates the R-squared (coefficient of determination) score between the actual and predicted values on the test set.



