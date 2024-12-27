import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train SVM model."""
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning for Random Forest using GridSearchCV."""
    model = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def model_selection(X_train, y_train):
    """Select and train multiple models."""
    # Train models
    models = {
        'Logistic Regression': train_logistic_regression(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'SVM': train_svm(X_train, y_train)
    }
    
    # Optionally, tune Random Forest
    best_rf_model = tune_random_forest(X_train, y_train)
    models['Tuned Random Forest'] = best_rf_model
    
    return models

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    return accuracy, conf_matrix, class_report

