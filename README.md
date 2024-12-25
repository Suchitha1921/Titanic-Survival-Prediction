Titanic Survival Prediction

Introduction  

The Titanic disaster is one of the most infamous maritime tragedies in history. Using a dataset containing details about passengers, this project aims to build a predictive model to determine whether a passenger survived the disaster. This task involves data preprocessing, feature selection, and applying machine learning techniques to develop an accurate and reliable classification model.


Dataset Description  

The dataset includes the following features:  
- PassengerId: Unique identifier for each passenger.  
- Survived: Target variable (0 = No, 1 = Yes).  
- Pclass: Ticket class (1st, 2nd, 3rd).  
- Name: Passenger's name.  
- Sex: Gender.  
- Age: Age in years.  
- SibSp: Number of siblings/spouses aboard.  
- Parch: Number of parents/children aboard.  
- Ticket: Ticket number.  
- Fare: Passenger fare.  
- Cabin: Cabin number.  
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).  


Objective  

The objective is to predict the survival of passengers using the given features while maintaining a high level of accuracy. The solution will be evaluated based on:  
1. Accuracy: Correctness of predictions.  
2. Approach: Soundness of preprocessing, feature selection, and model building.  
3. Documentation: Clarity and completeness of the explanation.  


Approach  

Step 1: Data Exploration  
- Loaded the dataset and analyzed it to understand the distribution of features.  
- Visualized relationships between features and the target variable using plots.  

Step 2: Data Preprocessing  
- Missing Values:  
  - Imputed missing values for Age using median values.  
  - Dropped Cabin due to a high percentage of missing values.  
  - Filled missing values in Embarked with the mode.  
- Encoding Categorical Variables:  
  - Converted Sex and Embarked into numerical values using one-hot encoding.  
- Feature Engineering:  
  - Created a new feature, FamilySize, by summing SibSp and Parch.  

Step 3: Feature Selection  
- Retained relevant features such as Pclass, Sex, Age, Fare, FamilySize, and Embarked.  
- Dropped less impactful features like Name and Ticket.  

Step 4: Model Building  
- Algorithms Used:  
  - Logistic Regression.  
  - Random Forest Classifier.  
  - Gradient Boosting (XGBoost).  
- Evaluation Metrics:  
  - Accuracy.  
  - Precision, Recall, and F1-score.  
  - ROC-AUC score.  

Step 5: Hyperparameter Tuning  
- Optimized hyperparameters for the Random Forest and XGBoost models using GridSearchCV.  

Step 6: Model Evaluation  
- Evaluated the models on the test dataset.  
- Selected the best model based on accuracy and ROC-AUC score.  


Challenges Faced  

1. Handling Missing Values: The Age and Cabin columns required careful imputation and decisions on data removal.  
2. Feature Engineering: Identifying impactful features without overfitting.  
3. Model Selection: Balancing accuracy and interpretability in model choice.  


Results  

- Best Model: Random Forest Classifier.  
- Accuracy: 85.3% on the test dataset.  
- Feature Importance: Key contributors included Sex, Pclass, and Fare.  

Repository Structure  

Titanic-Survival-Prediction/  
- data/  
  - raw/ (Raw dataset)  
  - processed/ (Preprocessed dataset)  
- notebooks/ (Jupyter notebooks for data exploration and modeling)  
- scripts/ (Python scripts for data preprocessing and modeling)  
- models/ (Saved models)  
- README.md (Project documentation)  


How to Run the Project 
 
1. Clone the repository:  
   git clone <repository_url>  
2. Install dependencies:  
   pip install -r requirements.txt  
3. Run the preprocessing script:  
   python scripts/preprocess_data.py  
4. Train the model:  
   python scripts/train_model.py  
5. Evaluate the model:  
   python scripts/evaluate_model.py 


Conclusion  

This project demonstrates a comprehensive approach to predicting Titanic survival outcomes. By leveraging data preprocessing, feature engineering, and robust modeling techniques, the solution achieves high accuracy and offers insights into factors influencing survival.  


