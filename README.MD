# Customer Queries Classification

This project focuses on classifying customer queries into various predefined categories using machine learning techniques.

## Project Structure
The project directory is organized as follows:
```
classification_lightGBM
│
├── config
│ └── config.yaml # Configuration file with paths and parameters
│
├── data
│ ├── customer_queries_data.csv # Training data
│ ├── customer_queries_test.csv # Test data for evaluation
│ └── inference_output.csv # Output predictions
│
├── model
│ ├── classification_report.txt # Generated classification report
│ ├── confusion_matrix.png # Generated confusion matrix
│ ├── label_encoder.pkl # Saved label encoder
│ ├── lgbm_model.pkl # Trained LightGBM model
│ ├── parameters.yaml # Model parameters
│ └── vectorizer.pkl # Saved TF-IDF vectorizer
│
├── notebooks
│ └── EDA.ipynb # Jupyter notebook for Exploratory Data Analysis
│
├── src
│ ├── data_preparation.py # Script for data preparation
│ ├── inference.py # Script for running inference
│ ├── train.py # Script for training the model
│ └── utils.py # Utility functions
│
├── README.MD # Project README file
├── Report.pdf # Report file
└── requirements.txt # List of Python packages required for the project
```

## Usage
- Install the required Python packages
- Train the model
- Run inference

## Installation
To install the required Python packages, use the following command:
```
pip install -r requirements.txt
```

## Configuration
The config/config.yaml file contains paths to data files and model components, as well as other configuration parameters.

## Training the Model
To train the model, execute the train.py script:
```
python src/train.py
```
This script will:
- Prepare the training data.
- Train the LightGBM model using the specified parameters.
- Evaluate the model on the test data.
- Save the trained model, label encoder, TF-IDF vectorizer, and evaluation results.

## Running Inference
To run inference and predict labels for new data, execute the inference.py script:
```
python src/inference.py
```
This script will:
- Load the trained model, vectorizer, and label encoder.
- Prepare the inference data.
- Make predictions on the inference data.
- Save the predictions to a CSV file.

## Exploratory Data Analysis (EDA)
The notebooks/EDA.ipynb Jupyter notebook contains exploratory data analysis to understand the distribution of data, query lengths, and class imbalances.
It provides insights that help in making decisions for preprocessing and model training.

## Results
The evaluation results include a classification report and a confusion matrix, which are saved in the model directory:
- classification_report.txt: Contains precision, recall, and F1-score for each class.
- confusion_matrix.png: Visual representation of the confusion matrix for the test set.

## Conclusion
This project demonstrates a complete workflow for training a machine learning model to classify customer queries.
The provided scripts and configuration files make it easy to reproduce the results and adapt the project to similar tasks.