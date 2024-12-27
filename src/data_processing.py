import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Load Titanic dataset from CSV file."""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Fill missing values for Age with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fill missing Embarked with the mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Drop rows with missing Cabin and Fare
    df.drop(columns=['Cabin'], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    return df

def encode_categorical_features(df):
    """Encode categorical features using LabelEncoder."""
    label_encoder = LabelEncoder()
    
    # Encoding 'Sex' and 'Embarked'
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    
    return df

def scale_features(df):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    return df

def preprocess_data(df):
    """Complete preprocessing pipeline."""
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    df = scale_features(df)
    return df

def split_data(df, target_column='Survived', test_size=0.2, random_state=42):
    """Split the data into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

