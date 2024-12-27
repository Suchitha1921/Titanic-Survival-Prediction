import pandas as pd

def create_title_feature(df):
    """Extract title from the Name column and create a new 'Title' feature."""
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    # Map rare titles to 'Other'
    rare_titles = ['Dr', 'Rev', 'Major', 'Lady', 'Sir', 'Capt', 'Countess', 'Jonkheer', 'Don', 'Dona']
    df['Title'] = df['Title'].apply(lambda x: x if x not in rare_titles else 'Other')
    
    # One-hot encode Title
    df = pd.get_dummies(df, columns=['Title'], drop_first=True)
    
    return df

def feature_selection(df):
    """Select relevant features for the model."""
    # Drop 'Name' as it's not a useful feature for prediction
    df.drop(columns=['Name', 'Ticket'], inplace=True)
    
    return df

def feature_engineering(df):
    """Complete feature engineering pipeline."""
    df = create_title_feature(df)
    df = feature_selection(df)
    return df

