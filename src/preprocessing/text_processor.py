"""
Text preprocessing utilities for student feedback analysis.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextProcessor:
    """
    Class for preprocessing text data from student feedback.
    """
    
    def __init__(self, 
                 remove_stopwords=True, 
                 remove_punctuation=True,
                 lemmatize=True,
                 stem=False,
                 lowercase=True):
        """
        Initialize the text processor with specified options.
        
        Parameters:
        -----------
        remove_stopwords : bool
            Whether to remove stopwords
        remove_punctuation : bool
            Whether to remove punctuation
        lemmatize : bool
            Whether to lemmatize words
        stem : bool
            Whether to stem words (not recommended to use both lemmatize and stem)
        lowercase : bool
            Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize
        self.stem = stem
        self.lowercase = lowercase
        
        # Initialize components
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        if stem:
            self.stemmer = PorterStemmer()
    
    def preprocess(self, text):
        """
        Preprocess a single text document.
        
        Parameters:
        -----------
        text : str
            The text to preprocess
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if specified
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Stem if specified
        if self.stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Join tokens back into a string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess text in a DataFrame column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the text data
        text_column : str
            Name of the column containing text to preprocess
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with an additional column 'processed_text'
        """
        df = df.copy()
        df['processed_text'] = df[text_column].apply(self.preprocess)
        return df


def extract_features(texts, method='tfidf', max_features=5000, ngram_range=(1, 2)):
    """
    Extract features from preprocessed text.
    
    Parameters:
    -----------
    texts : list or pandas.Series
        Collection of preprocessed text documents
    method : str
        Feature extraction method ('tfidf', 'count', or 'binary')
    max_features : int
        Maximum number of features to extract
    ngram_range : tuple
        Range of n-grams to consider
        
    Returns:
    --------
    scipy.sparse.csr_matrix, sklearn.feature_extraction.text.TfidfVectorizer
        Feature matrix and the fitted vectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, 
                                     ngram_range=ngram_range)
    elif method == 'count':
        vectorizer = CountVectorizer(max_features=max_features, 
                                     ngram_range=ngram_range)
    elif method == 'binary':
        vectorizer = CountVectorizer(max_features=max_features, 
                                     ngram_range=ngram_range,
                                     binary=True)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'tfidf', 'count', or 'binary'.")
    
    feature_matrix = vectorizer.fit_transform(texts)
    
    return feature_matrix, vectorizer
