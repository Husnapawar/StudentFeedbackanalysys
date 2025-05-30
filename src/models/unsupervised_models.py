"""
Unsupervised learning models for student feedback analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

class TopicModeler:
    """
    Topic modeling for student feedback using LDA or NMF.
    """
    
    def __init__(self, method='lda', n_topics=5, max_features=5000, random_state=42):
        """
        Initialize the topic modeler.
        
        Parameters:
        -----------
        method : str
            Method to use ('lda' or 'nmf')
        n_topics : int
            Number of topics to extract
        max_features : int
            Maximum number of features to use
        random_state : int
            Random seed for reproducibility
        """
        self.method = method
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on method."""
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=10,
                learning_method='online',
                random_state=self.random_state
            )
        elif self.method == 'nmf':
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                alpha=0.1,
                l1_ratio=0.5
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'lda' or 'nmf'.")
        
        self.vectorizer = CountVectorizer(max_features=self.max_features)
    
    def fit(self, texts):
        """
        Fit the topic model to the data.
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Collection of preprocessed text documents
        """
        self.document_term_matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(self.document_term_matrix)
    
    def transform(self, texts):
        """
        Transform new texts to topic distributions.
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Collection of preprocessed text documents
            
        Returns:
        --------
        array
            Topic distributions for each document
        """
        dtm = self.vectorizer.transform(texts)
        return self.model.transform(dtm)
    
    def get_top_words_per_topic(self, n_words=10):
        """
        Get the top words for each topic.
        
        Parameters:
        -----------
        n_words : int
            Number of top words to return per topic
            
        Returns:
        --------
        list of list
            List of top words for each topic
        """
        feature_names = self.vectorizer.get_feature_names_out()
        
        top_words = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words.append([feature_names[i] for i in top_indices])
        
        return top_words
    
    def get_topic_distribution(self, texts):
        """
        Get the topic distribution for each document.
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Collection of preprocessed text documents
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with topic distributions for each document
        """
        topic_distribution = self.transform(texts)
        
        # Create a DataFrame with topic distributions
        columns = [f'Topic_{i+1}' for i in range(self.n_topics)]
        df_topics = pd.DataFrame(topic_distribution, columns=columns)
        
        # Add dominant topic column
        df_topics['Dominant_Topic'] = df_topics.idxmax(axis=1)
        
        return df_topics


class FeedbackClusterer:
    """
    Clustering for student feedback.
    """
    
    def __init__(self, method='kmeans', n_clusters=5, random_state=42):
        """
        Initialize the feedback clusterer.
        
        Parameters:
        -----------
        method : str
            Clustering method ('kmeans', 'hierarchical', or 'dbscan')
        n_clusters : int
            Number of clusters (not used for DBSCAN)
        random_state : int
            Random seed for reproducibility
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on method."""
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.method == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                affinity='euclidean',
                linkage='ward'
            )
        elif self.method == 'dbscan':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'kmeans', 'hierarchical', or 'dbscan'.")
    
    def fit(self, X):
        """
        Fit the clustering model to the data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
        """
        self.model.fit(X)
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
            
        Returns:
        --------
        array
            Predicted cluster labels
        """
        if self.method == 'kmeans':
            return self.model.predict(X)
        elif self.method == 'hierarchical':
            # AgglomerativeClustering doesn't have a predict method
            # We need to fit again with the new data
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                affinity='euclidean',
                linkage='ward'
            )
            return model.fit_predict(X)
        elif self.method == 'dbscan':
            # DBSCAN doesn't have a predict method
            # We need to fit again with the new data
            model = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
            return model.fit_predict(X)
    
    def evaluate(self, X):
        """
        Evaluate the clustering model.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        labels = self.model.labels_ if hasattr(self.model, 'labels_') else self.model.predict(X)
        
        # Calculate silhouette score if there are at least 2 clusters and not all points are in the same cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(X):
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = None
        
        return {
            'silhouette_score': silhouette,
            'n_clusters': len(unique_labels),
            'cluster_sizes': {label: np.sum(labels == label) for label in unique_labels}
        }
    
    def get_cluster_distribution(self, X):
        """
        Get the cluster distribution for the data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster assignments
        """
        labels = self.model.labels_ if hasattr(self.model, 'labels_') else self.model.predict(X)
        
        return pd.DataFrame({'Cluster': labels})


class DimensionalityReducer:
    """
    Dimensionality reduction for visualization and feature extraction.
    """
    
    def __init__(self, method='svd', n_components=2, random_state=42):
        """
        Initialize the dimensionality reducer.
        
        Parameters:
        -----------
        method : str
            Method to use ('svd', 'nmf', or 'lda')
        n_components : int
            Number of components to extract
        random_state : int
            Random seed for reproducibility
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on method."""
        if self.method == 'svd':
            self.model = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'nmf':
            self.model = NMF(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_components,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'svd', 'nmf', or 'lda'.")
    
    def fit(self, X):
        """
        Fit the dimensionality reduction model to the data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
        """
        self.model.fit(X)
    
    def transform(self, X):
        """
        Transform data to lower-dimensional space.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
            
        Returns:
        --------
        array
            Transformed data
        """
        return self.model.transform(X)
    
    def fit_transform(self, X):
        """
        Fit the model and transform the data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
            
        Returns:
        --------
        array
            Transformed data
        """
        return self.model.fit_transform(X)
