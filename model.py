import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data(file_path):
    columns = [
        "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
        "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
        "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
        "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
        "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
        "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
        "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", 
        "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
        "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
        "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
        "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
        "capital_run_length_longest", "capital_run_length_total", "label"
    ]
    return pd.read_csv(file_path, header=None, names=columns)

def train_model(df):
    X = df.drop(columns=['label'])  # Features
    y = df['label']  # Target variable
    
    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 3: Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 1: Model Selection - Logistic Regression
    model = LogisticRegression()

    # Step 2: Model Training
    model.fit(X_train_scaled, y_train)

    # Step 3: Model Evaluation
    y_pred = model.predict(X_test_scaled)
    
    # Print the confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and scaler to files
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future use

    return model, scaler, X_test_scaled, y_test  # Ensure these are returned

def load_model():
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler
