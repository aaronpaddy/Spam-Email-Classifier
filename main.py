import os
import joblib
from preprocess import preprocess_email
from model import load_data, train_model

def main():
    # Get the current directory of the main.py script
    base_dir = os.path.dirname(__file__)
    
    # Dynamically create the path for spambase.data using the base directory
    file_path = os.path.join(base_dir, 'spambase.data')

    # Load the dataset
    df = load_data(file_path)

    # Train the model and get model, scaler, and test data
    model, scaler, X_test_scaled, y_test = train_model(df)

    # Dynamically create the paths for the model and scaler files
    model_path = os.path.join(base_dir, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')

    # Save the model and scaler to dynamically constructed paths
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Load the model and scaler for future predictions
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Prompt the user to enter the email
    print("Please enter the email you want to check (end with an empty line):")

    # Initialize an empty list to hold each line of the email
    email_lines = []

    # Loop to collect multiple lines of email
    while True:
        line = input()  # Read each line
        if line:  # If the line is not empty
            email_lines.append(line)  # Add the line to the list
        else:  # If the line is empty
            break  # Exit the loop

    # Join the lines into a single string for processing
    new_email = "\n".join(email_lines)

    # Preprocess the email
    preprocessed_email = preprocess_email(new_email)

    # Scale the new email features using the same scaler as your training data
    scaled_email = scaler.transform(preprocessed_email)

    # Predict if it's spam (1) or not spam (0)
    prediction = model.predict(scaled_email)

    if prediction == 1:
        print("This email is classified as SPAM.")
    else:
        print("This email is classified as NOT SPAM.")

if __name__ == "__main__":
    main()
