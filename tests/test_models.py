"""
Unit tests for the machine learning models.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.text_processor import TextProcessor, extract_features
from src.models.supervised_models import SentimentClassifier, FeedbackCategorizer
from src.models.unsupervised_models import TopicModeler, FeedbackClusterer, DimensionalityReducer

class TestTextProcessor(unittest.TestCase):
    """Test cases for the TextProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.text_processor = TextProcessor(
            remove_stopwords=True,
            remove_punctuation=True,
            lemmatize=True,
            stem=False,
            lowercase=True
        )
        self.sample_texts = [
            "I really enjoyed the Mathematics course. The teaching quality was excellent.",
            "The assessment methods in Physics need significant improvement.",
            "Neither good nor bad support services in the Literature course."
        ]
    
    def test_preprocess(self):
        """Test text preprocessing."""
        processed_text = self.text_processor.preprocess(self.sample_texts[0])
        self.assertIsInstance(processed_text, str)
        self.assertTrue(len(processed_text) > 0)
        
        # Check if stopwords are removed
        self.assertNotIn("the", processed_text)
        self.assertNotIn("was", processed_text)
        
        # Check if text is lowercased
        self.assertNotIn("Mathematics", processed_text)
        self.assertIn("mathematics", processed_text)
        
        # Check if punctuation is removed
        self.assertNotIn(".", processed_text)
    
    def test_preprocess_dataframe(self):
        """Test preprocessing a DataFrame."""
        df = pd.DataFrame({'text': self.sample_texts})
        processed_df = self.text_processor.preprocess_dataframe(df, 'text')
        
        self.assertIn('processed_text', processed_df.columns)
        self.assertEqual(len(processed_df), len(self.sample_texts))
        self.assertTrue(all(isinstance(text, str) for text in processed_df['processed_text']))

class TestFeatureExtraction(unittest.TestCase):
    """Test cases for feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "I really enjoyed the Mathematics course. The teaching quality was excellent.",
            "The assessment methods in Physics need significant improvement.",
            "Neither good nor bad support services in the Literature course.",
            "Great course content in Computer Science, very helpful for my learning.",
            "I was disappointed with the facilities in the Chemistry course."
        ]
    
    def test_extract_features_tfidf(self):
        """Test TF-IDF feature extraction."""
        feature_matrix, vectorizer = extract_features(
            self.sample_texts,
            method='tfidf',
            max_features=100,
            ngram_range=(1, 2)
        )
        
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertEqual(feature_matrix.shape[0], len(self.sample_texts))
        self.assertTrue(feature_matrix.shape[1] <= 100)  # Max features
    
    def test_extract_features_count(self):
        """Test count vectorizer feature extraction."""
        feature_matrix, vectorizer = extract_features(
            self.sample_texts,
            method='count',
            max_features=100,
            ngram_range=(1, 2)
        )
        
        self.assertEqual(feature_matrix.shape[0], len(self.sample_texts))
        self.assertTrue(feature_matrix.shape[1] <= 100)  # Max features

class TestSupervisedModels(unittest.TestCase):
    """Test cases for supervised learning models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.X_train = np.random.rand(100, 50)  # 100 samples, 50 features
        self.y_sentiment = np.random.choice(['Positive', 'Neutral', 'Negative'], size=100)
        self.y_category = np.random.choice(['Course Content', 'Teaching Quality', 'Assessment Methods'], size=100)
    
    def test_sentiment_classifier(self):
        """Test sentiment classifier."""
        for model_type in ['logistic_regression', 'naive_bayes', 'random_forest', 'svm']:
            classifier = SentimentClassifier(model_type=model_type)
            classifier.train(self.X_train, self.y_sentiment)
            
            # Test prediction
            X_test = np.random.rand(10, 50)
            predictions = classifier.predict(X_test)
            
            self.assertEqual(len(predictions), 10)
            for pred in predictions:
                self.assertIn(pred, ['Positive', 'Neutral', 'Negative'])
    
    def test_feedback_categorizer(self):
        """Test feedback categorizer."""
        for model_type in ['logistic_regression', 'naive_bayes', 'random_forest', 'svm']:
            categorizer = FeedbackCategorizer(model_type=model_type)
            categorizer.train(self.X_train, self.y_category)
            
            # Test prediction
            X_test = np.random.rand(10, 50)
            predictions = categorizer.predict(X_test)
            
            self.assertEqual(len(predictions), 10)
            for pred in predictions:
                self.assertIn(pred, ['Course Content', 'Teaching Quality', 'Assessment Methods'])

class TestUnsupervisedModels(unittest.TestCase):
    """Test cases for unsupervised learning models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_texts = [
            "I really enjoyed the Mathematics course. The teaching quality was excellent.",
            "The assessment methods in Physics need significant improvement.",
            "Neither good nor bad support services in the Literature course.",
            "Great course content in Computer Science, very helpful for my learning.",
            "I was disappointed with the facilities in the Chemistry course."
        ] * 5  # Repeat to get more samples
        
        # Extract features
        self.feature_matrix, self.vectorizer = extract_features(
            self.sample_texts,
            method='tfidf',
            max_features=100
        )
    
    def test_topic_modeler(self):
        """Test topic modeling."""
        for method in ['lda', 'nmf']:
            topic_modeler = TopicModeler(method=method, n_topics=3)
            topic_modeler.fit(self.sample_texts)
            
            # Test getting top words
            top_words = topic_modeler.get_top_words_per_topic(n_words=5)
            self.assertEqual(len(top_words), 3)  # 3 topics
            for words in top_words:
                self.assertEqual(len(words), 5)  # 5 words per topic
            
            # Test topic distribution
            topic_dist = topic_modeler.get_topic_distribution(self.sample_texts)
            self.assertEqual(len(topic_dist), len(self.sample_texts))
    
    def test_clusterer(self):
        """Test clustering."""
        for method in ['kmeans', 'hierarchical']:
            clusterer = FeedbackClusterer(method=method, n_clusters=3)
            clusterer.fit(self.feature_matrix)
            
            # Test prediction
            predictions = clusterer.predict(self.feature_matrix)
            self.assertEqual(len(predictions), len(self.feature_matrix))
            
            # Test evaluation
            eval_results = clusterer.evaluate(self.feature_matrix)
            self.assertIn('silhouette_score', eval_results)
            self.assertIn('n_clusters', eval_results)
    
    def test_dimensionality_reducer(self):
        """Test dimensionality reduction."""
        for method in ['svd', 'nmf']:
            reducer = DimensionalityReducer(method=method, n_components=10)
            reducer.fit(self.feature_matrix)
            
            # Test transformation
            reduced = reducer.transform(self.feature_matrix)
            self.assertEqual(reduced.shape[0], self.feature_matrix.shape[0])
            self.assertEqual(reduced.shape[1], 10)

if __name__ == '__main__':
    unittest.main()
