import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
def load_file(file_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
# Train Logistic Regression Model
def train_model(df):
    """Train a Logistic Regression model."""
    try:
        # Select the features (X) and target (y)
        X = df[["Age", "Department_Encoded", "JobRole_Encoded", "YearsAtCompany", "MonthlyIncome", 
                "Gender_Encoded", "Overtime_Encoded"]]
        
        y = df["Attrition_Encoded"]
      
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Save the trained model to system
        joblib.dump(model, "model.pkl")

        # Predict on test data
        y_pred = model.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate training accuracy
        y_train_pred = model.predict(X_train_scaled)
        training_accuracy = accuracy_score(y_train, y_train_pred)

        # Calculate testing accuracy
        y_test_pred = model.predict(X_test_scaled)
        testing_accuracy = accuracy_score(y_test, y_test_pred)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return model, accuracy, training_accuracy, testing_accuracy
    
    except Exception as e:
        print("Error during training:", str(e))
        return None

# Main execution
if __name__ == "__main__":
    file_path = "cleaned_hr_data.csv"  # Path to your dataset

    df = load_file(file_path)
    if df is not None:
        # Train and save the model
        model, accuracy, training_accuracy, testing_accuracy = train_model(df)
        print(f"Model Accuracy : {accuracy}")
        print(f"Training Accuracy : {training_accuracy}")
        print(f"Testing Accuracy : {testing_accuracy}")
    