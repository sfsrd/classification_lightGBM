from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_yaml, save_object, save_classification_report
from data_preparation import DataPreparator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelTrainer:
    """
    Class for training and evaluating a LightGBM model.

    This class handles preparing the data, training the model, evaluating it,
    and saving the model and results.
    """

    def __init__(self, config_path):
        """
        Initialize the ModelTrainer with configuration settings.

        :param config_path: Path to the configuration YAML file.
        """
        self.config = load_yaml(config_path)
        self.data_preparator = DataPreparator(self.config)
        self.model_parameters = self.config['model_parameters_path']
        self.model_path = self.config['model_path']
        self.report_path = self.config['report_path']
        self.confusion_matrix_path = self.config['confusion_matrix_path']

    def prepare_data(self):
        """
        Prepare the training and test data.

        :return: X_train, X_test, y_train, y_test, label_encoder
        """
        X, y, label_encoder = self.data_preparator.prepare_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, label_encoder

    def train_model(self, X_train, y_train, params):
        """
        Train the LightGBM model with the given parameters.

        :param X_train: Training data features.
        :param y_train: Training data labels.
        :param params: Model parameters.
        :return: Trained model.
        """
        lgb_model = lgb.LGBMClassifier(**params)
        logging.info("Training the model...")
        lgb_model.fit(X_train, y_train)
        return lgb_model

    def get_classification_report(self, y_test, y_test_pred, target_names):
        """
        Generate and save the classification report.

        :param y_test: True labels for the test set.
        :param y_test_pred: Predicted labels for the test set.
        :param target_names: Names of the target classes.
        :return: Classification report.
        """
        classification_rep = classification_report(y_test, y_test_pred, target_names=target_names)
        save_classification_report(classification_rep, self.report_path)
        return classification_rep

    def get_confusion_matrix(self, y_test, y_test_pred, target_names):
        """
        Generate and save the confusion matrix.

        :param y_test: True labels for the test set.
        :param y_test_pred: Predicted labels for the test set.
        :param target_names: Names of the target classes.
        """
        cm_test = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(18, 10))
        sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('LightGBM Test Set Confusion Matrix')
        plt.xticks(rotation=40, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3, left=0.2)
        plt.savefig(self.confusion_matrix_path)

    def evaluate_model(self, model, X_test, y_test, label_encoder):
        """
        Evaluate the model on the test set and generate reports.

        :param model: Trained model.
        :param X_test: Test data features.
        :param y_test: Test data labels.
        :param label_encoder: Label encoder used for decoding the labels.
        """
        y_test_pred = model.predict(X_test)
        classification_rep = self.get_classification_report(y_test, y_test_pred, target_names=label_encoder.classes_)
        print("Test Set Classification Report:\n", classification_rep)
        self.get_confusion_matrix(y_test, y_test_pred, target_names=label_encoder.classes_)

    def save_model(self, model):
        """
        Save the trained model to the specified path.

        :param model: Trained model.
        """
        save_object(model, self.model_path)

    def run(self):
        """
        Execute the training and evaluation process.
        """
        X_train, X_test, y_train, y_test, label_encoder = self.prepare_data()
        params = load_yaml(self.model_parameters)
        model = self.train_model(X_train, y_train, params)
        self.evaluate_model(model, X_test, y_test, label_encoder)
        self.save_model(model)


if __name__ == '__main__':
    config_path = '../config/config.yaml'
    trainer = ModelTrainer(config_path)
    trainer.run()
