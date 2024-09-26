
## Spam Email Classification Project

This project aims to classify emails as either **spam** or **not spam** using a machine learning model trained on the [UCI Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase).

## Introduction

This project classifies emails as spam or not spam using a **Logistic Regression** model. The data comes from the UCI Spambase dataset, which includes features extracted from emails, such as word frequencies and the frequency of specific characters.


## Installation

### Prerequisites

- Python 3.x
- Required Python libraries are listed in `requirements.txt`

### Steps to Set Up

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aaronpaddy/Spam-Email-Classifier.git
   cd spam-email-classifier
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate    # On Windows: env\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**: Ensure that the `spambase.data` file is in the project root directory (or adjust the path accordingly in `main.py`).

## Usage

1. **Run the project**:
   To train the model and classify new emails, run the `main.py` script:

   ```bash
   python main.py
   ```

2. **Interactive Email Classification**:
   After running the script, you will be prompted to input an email for classification. Enter the email content, and when done, submit an empty line to see if it's classified as spam or not.

## Model Details

- The model is a **Logistic Regression** classifier.
- The data is preprocessed by scaling the features using `StandardScaler`.
- The model is trained on the UCI Spambase dataset and evaluated using a test split.

## Preprocessing

Before classifying an email, it is processed to extract relevant features such as word frequencies and character occurrences. The `preprocess.py` file contains the code to transform new email inputs into the correct format for the trained model.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

1. Fork the project
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

## License

This project is licensed under the MIT License.
```

