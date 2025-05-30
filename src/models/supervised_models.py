"""
Supervised learning models for student feedback analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

class SentimentClassifier:
    """
    Classifier for sentiment analysis of student feedback.
    """
    
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize the sentiment classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic_regression', 'naive_bayes', 
            'random_forest', or 'svm')
        """
        self.model_type = model_type
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, C=1.0, 
                                           class_weight='balanced')
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, 
                                               class_weight='balanced')
        elif self.model_type == 'svm':
            self.model = LinearSVC(C=1.0, class_weight='balanced', 
                                  max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the sentiment classifier.
        
        Parameters:
        -----------
        X_train : array-like or sparse matrix
            Training data features
        y_train : array-like
            Target sentiment labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Predict sentiment labels for new data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
            
        Returns:
        --------
        array
            Predicted sentiment labels
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : array-like or sparse matrix
            Test data features
        y_test : array-like
            True sentiment labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5):
        """
        Tune hyperparameters using grid search.
        
        Parameters:
        -----------
        X_train : array-like or sparse matrix
            Training data features
        y_train : array-like
            Target sentiment labels
        param_grid : dict, optional
            Grid of hyperparameters to search
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Best parameters found
        """
        if param_grid is None:
            # Default parameter grids for each model type
            if self.model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'saga']
                }
            elif self.model_type == 'naive_bayes':
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'loss': ['hinge', 'squared_hinge']
                }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_


class FeedbackCategorizer:
    """
    Multi-class classifier for categorizing student feedback.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the feedback categorizer.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic_regression', 'naive_bayes', 
            'random_forest', or 'svm')
        """
        self.model_type = model_type
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, C=1.0, 
                                           multi_class='multinomial',
                                           class_weight='balanced')
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, 
                                               class_weight='balanced')
        elif self.model_type == 'svm':
            self.model = LinearSVC(C=1.0, class_weight='balanced', 
                                  max_iter=1000, multi_class='ovr')
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the feedback categorizer.
        
        Parameters:
        -----------
        X_train : array-like or sparse matrix
            Training data features
        y_train : array-like
            Target category labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Predict category labels for new data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            Data features
            
        Returns:
        --------
        array
            Predicted category labels
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : array-like or sparse matrix
            Test data features
        y_test : array-like
            True category labels
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None, cv=5):
        """
        Tune hyperparameters using grid search.
        
        Parameters:
        -----------
        X_train : array-like or sparse matrix
            Training data features
        y_train : array-like
            Target category labels
        param_grid : dict, optional
            Grid of hyperparameters to search
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Best parameters found
        """
        if param_grid is None:
            # Default parameter grids for each model type
            if self.model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['lbfgs', 'saga']
                }
            elif self.model_type == 'naive_bayes':
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'loss': ['hinge', 'squared_hinge']
                }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
