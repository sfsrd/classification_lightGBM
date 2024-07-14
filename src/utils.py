import yaml
import logging
import pandas as pd
import joblib
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yaml(file_path):
    """
    Read a YAML file and return its contents as a dictionary.

    :param file_path: Path to the YAML file.
    :return: Contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            logging.error(f"Error reading YAML file: {exc}")
            return None


def read_csv(file_path):
    """
    Read a CSV file and return its contents as a DataFrame.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the contents of the CSV file, or None if an error occurs.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully read CSV file: {file_path}")
        return data
    except Exception as exc:
        logging.error(f"An error occurred while reading the CSV file: {exc}")
        return None

def save_csv(data, file_path):
    """
    Save a DataFrame to a CSV file.

    :param data: DataFrame to be saved.
    :param file_path: Path to the CSV file.
    :return: None
    """
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Successfully saved CSV file: {file_path}")
    except Exception as exc:
        logging.error(f"An error occurred while saving the CSV file: {exc}")

def load_object(file_path):
    """
    Load any object from a file using joblib.

    :param file_path: Path to the file.
    :return: Loaded object.
    """
    try:
        obj = joblib.load(file_path)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        return None


def save_object(obj, file_path):
    """
    Save any object to a file using joblib.

    :param obj: Object to be saved.
    :param file_path: Path to the file where the object will be saved.
    :return: None
    """
    try:
        joblib.dump(obj, file_path)
        logging.info(f"Object {obj} saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object {obj} to {file_path}: {e}")


def save_classification_report(report, report_path):
    """
    Save the classification report to a file.

    :return: None
    """
    with open(report_path, 'w') as f:
        f.write("Test Set Classification Report:\n")
        f.write(report)
    logging.info(f"Classification report saved as {report_path}")