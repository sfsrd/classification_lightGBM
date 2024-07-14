import pandas as pd
from data_preparation import DataPreparator
from utils import load_yaml, save_csv, load_object
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelInference:
    """
    Class for performing inference using a trained model.

    This class handles loading the trained model, preparing inference data, making predictions,
    and saving the results.
    """

    def __init__(self, config_path):
        """
        Initialize the ModelInference with configuration settings.

        :param config_path: Path to the configuration YAML file.
        """
        self.config = load_yaml(config_path)
        self.model_path = self.config['model_path']
        self.vectorizer_path = self.config['vectorizer_path']
        self.label_encoder_path = self.config['label_encoder_path']
        self.inference_data_path = self.config['inference_data_file_path']
        self.output_data_path = self.config['output_data_file_path']
        self.report_path = self.config['report_path']
        self.model = None
        self.vectorizer = None
        self.label_encoder = None

    def load_model(self):
        """
        Load the trained model, vectorizer, and label encoder from their respective paths.
        """
        self.model = load_object(self.model_path)
        self.vectorizer = load_object(self.vectorizer_path)
        self.label_encoder = load_object(self.label_encoder_path)

    def prepare_data(self):
        """
        Prepare the inference data by reading the data and preprocessing it using the provided vectorizer.

        :return: The vectorized inference data and the original queries.
        """
        data_preparator = DataPreparator(self.config)
        X_inference, original_queries = data_preparator.prepare_inference_data(self.vectorizer)
        return X_inference, original_queries

    def predict(self, X_inference):
        """
        Make predictions on the inference data using the trained model.

        :param X_inference: The vectorized inference data.
        :return: The predicted labels.
        """
        predictions = self.model.predict(X_inference)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        return predicted_labels

    def save_predictions(self, original_queries, predicted_labels):
        """
        Save the predictions to a CSV file.

        :param original_queries: The original queries.
        :param predicted_labels: The predicted labels.
        """
        result_df = pd.DataFrame({
            'query': original_queries,
            'predicted_label': predicted_labels
        })
        save_csv(result_df, self.output_data_path)

    def run(self):
        """
        Execute the inference process: load the model, prepare the data, make predictions, and save the results.
        """
        self.load_model()
        X_inference, original_queries = self.prepare_data()
        predictions = self.predict(X_inference)
        logging.info("Predictions done.")
        self.save_predictions(original_queries, predictions)


if __name__ == '__main__':
    config_path = '../config/config.yaml'
    inference = ModelInference(config_path)
    inference.run()
