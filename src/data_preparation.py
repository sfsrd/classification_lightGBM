import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
from utils import read_csv, save_object

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if NLTK data is downloaded, download if not present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

class DataPreparator:
    """
    Class for preparing data for training and inference.

    This class handles text preprocessing, vectorization, and label encoding for training and inference datasets.
    """

    def __init__(self, config):
        """
        Initialize the DataPreparator with configuration settings.

        :param config: Configuration dictionary containing paths for data files and model components.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.label_encoder = None
        self.train_data_path = config['train_data_file_path']
        self.inference_data_path = config['inference_data_file_path']
        self.vectorizer_path = config['vectorizer_path']
        self.label_encoder_path = config['label_encoder_path']

    def preprocess_text(self, text):
        """
        Preprocess the input text by removing special characters, converting to lowercase, tokenizing,
        removing stopwords, and lemmatizing.

        :param text: The input text to preprocess.
        :return: The preprocessed text.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def fit_transform_text(self, train_data):
        """
        Apply preprocessing and vectorization to the training data.

        :param train_data: The training data containing the 'query' column.
        :return: The vectorized training data and cleaned queries.
        """
        train_data['cleaned_query'] = train_data['query'].apply(self.preprocess_text)
        self.vectorizer = TfidfVectorizer()
        X_train = self.vectorizer.fit_transform(train_data['cleaned_query'])
        return X_train, train_data['cleaned_query']

    def encode_labels(self, labels):
        """
        Encode the labels using LabelEncoder.

        :param labels: The labels to encode.
        :return: The encoded labels.
        """
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        return y

    def save_prepared_data(self):
        """
        Save the vectorizer and label encoder to their respective paths.
        """
        save_object(self.vectorizer, self.vectorizer_path)
        save_object(self.label_encoder, self.label_encoder_path)

    def prepare_training_data(self):
        """
        Prepare the training data by reading the data, preprocessing, vectorizing, and encoding the labels.

        :return: The vectorized training data, encoded labels, and the label encoder.
        """
        train_data = read_csv(self.train_data_path)
        X, _ = self.fit_transform_text(train_data)
        y = self.encode_labels(train_data['label'])
        logging.info("Training Data prepared successfully.")
        self.save_prepared_data()
        return X, y, self.label_encoder

    def prepare_inference_data(self, vectorizer):
        """
        Prepare the inference data by reading the data and preprocessing it using the provided vectorizer.

        :param vectorizer: The fitted vectorizer to use for transforming the inference data.
        :return: The vectorized inference data and the original queries.
        """
        inference_data = read_csv(self.inference_data_path)
        original_queries = inference_data['query'].copy()
        inference_data['cleaned_query'] = inference_data['query'].apply(self.preprocess_text)
        X_inference = vectorizer.transform(inference_data['cleaned_query'])
        logging.info("Inference Data prepared successfully.")
        return X_inference, original_queries
