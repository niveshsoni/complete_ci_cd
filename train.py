# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (using the Iris dataset as an example)
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Train the model
def train_model():
    # Load data
    df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the model
    model = LogisticRegression(max_iter=200)

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model to a file
    joblib.dump(model, 'logistic_regression_model.pkl')
    print("Model saved as logistic_regression_model.pkl")

if __name__ == "__main__":
    train_model()
