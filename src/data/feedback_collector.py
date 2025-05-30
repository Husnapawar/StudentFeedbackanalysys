"""
Module for collecting and storing student feedback.
"""

import os
import pandas as pd
import json
from datetime import datetime

class FeedbackCollector:
    """
    Class for collecting and storing student feedback.
    """
    
    def __init__(self, storage_path='data/collected_feedback.csv'):
        """
        Initialize the feedback collector.
        
        Parameters:
        -----------
        storage_path : str
            Path to store collected feedback
        """
        self.storage_path = storage_path
        self.ensure_storage_exists()
    
    def ensure_storage_exists(self):
        """Ensure the storage file exists with proper headers."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.storage_path):
            headers = [
                'feedback_id', 'student_id', 'course_id', 'subject',
                'feedback_text', 'rating', 'date', 'category', 'sentiment'
            ]
            pd.DataFrame(columns=headers).to_csv(self.storage_path, index=False)
    
    def submit_feedback(self, feedback_data):
        """
        Submit new feedback.
        
        Parameters:
        -----------
        feedback_data : dict
            Dictionary containing feedback data with keys:
            - student_id (optional)
            - course_id (optional)
            - subject
            - feedback_text
            - rating
            - category (optional)
            
        Returns:
        --------
        dict
            Status of the submission
        """
        try:
            # Load existing feedback
            df = pd.read_csv(self.storage_path)
            
            # Generate feedback ID
            feedback_id = f"F{len(df) + 1:06d}"
            
            # Add required fields
            feedback_data['feedback_id'] = feedback_id
            feedback_data['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Set defaults for optional fields
            if 'student_id' not in feedback_data:
                feedback_data['student_id'] = 'anonymous'
            if 'course_id' not in feedback_data:
                feedback_data['course_id'] = 'unknown'
            if 'category' not in feedback_data:
                feedback_data['category'] = None
            if 'sentiment' not in feedback_data:
                feedback_data['sentiment'] = None
            
            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
            
            # Save updated dataframe
            df.to_csv(self.storage_path, index=False)
            
            return {
                'status': 'success',
                'message': 'Feedback submitted successfully',
                'feedback_id': feedback_id
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error submitting feedback: {str(e)}'
            }
    
    def get_feedback(self, limit=None, sort_by='date', ascending=False):
        """
        Get collected feedback.
        
        Parameters:
        -----------
        limit : int, optional
            Maximum number of feedback entries to return
        sort_by : str
            Column to sort by
        ascending : bool
            Sort order
            
        Returns:
        --------
        pandas.DataFrame
            Collected feedback
        """
        try:
            df = pd.read_csv(self.storage_path)
            
            # Sort
            df = df.sort_values(by=sort_by, ascending=ascending)
            
            # Limit
            if limit is not None:
                df = df.head(limit)
            
            return df
        
        except Exception as e:
            print(f"Error getting feedback: {str(e)}")
            return pd.DataFrame()
    
    def get_feedback_stats(self):
        """
        Get statistics about collected feedback.
        
        Returns:
        --------
        dict
            Statistics about collected feedback
        """
        try:
            df = pd.read_csv(self.storage_path)
            
            if len(df) == 0:
                return {
                    'total_count': 0,
                    'average_rating': None,
                    'subject_counts': {},
                    'category_counts': {},
                    'sentiment_counts': {}
                }
            
            # Calculate statistics
            stats = {
                'total_count': len(df),
                'average_rating': df['rating'].mean() if 'rating' in df.columns else None,
                'subject_counts': df['subject'].value_counts().to_dict() if 'subject' in df.columns else {},
                'category_counts': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
                'sentiment_counts': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {}
            }
            
            return stats
        
        except Exception as e:
            print(f"Error getting feedback stats: {str(e)}")
            return {
                'total_count': 0,
                'average_rating': None,
                'subject_counts': {},
                'category_counts': {},
                'sentiment_counts': {}
            }
    
    def predict_sentiment_and_category(self, feedback_text, sentiment_model=None, category_model=None):
        """
        Predict sentiment and category for feedback text.
        
        Parameters:
        -----------
        feedback_text : str
            Feedback text to analyze
        sentiment_model : object, optional
            Trained sentiment model
        category_model : object, optional
            Trained category model
            
        Returns:
        --------
        dict
            Predicted sentiment and category
        """
        from src.preprocessing.text_processor import TextProcessor, extract_features
        
        result = {}
        
        try:
            # Preprocess text
            text_processor = TextProcessor()
            processed_text = text_processor.preprocess(feedback_text)
            
            # Predict sentiment if model is provided
            if sentiment_model is not None:
                # Extract features
                feature_matrix, _ = extract_features([processed_text], method='tfidf')
                
                # Predict
                sentiment = sentiment_model.predict(feature_matrix)[0]
                result['sentiment'] = sentiment
            
            # Predict category if model is provided
            if category_model is not None:
                # Extract features
                feature_matrix, _ = extract_features([processed_text], method='tfidf')
                
                # Predict
                category = category_model.predict(feature_matrix)[0]
                result['category'] = category
        
        except Exception as e:
            print(f"Error predicting sentiment and category: {str(e)}")
        
        return result
    
    def update_feedback(self, feedback_id, updates):
        """
        Update existing feedback.
        
        Parameters:
        -----------
        feedback_id : str
            ID of the feedback to update
        updates : dict
            Dictionary containing fields to update
            
        Returns:
        --------
        dict
            Status of the update
        """
        try:
            # Load existing feedback
            df = pd.read_csv(self.storage_path)
            
            # Find the feedback
            mask = df['feedback_id'] == feedback_id
            if not mask.any():
                return {
                    'status': 'error',
                    'message': f'Feedback with ID {feedback_id} not found'
                }
            
            # Update fields
            for key, value in updates.items():
                if key in df.columns:
                    df.loc[mask, key] = value
            
            # Save updated dataframe
            df.to_csv(self.storage_path, index=False)
            
            return {
                'status': 'success',
                'message': f'Feedback {feedback_id} updated successfully'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error updating feedback: {str(e)}'
            }
    
    def delete_feedback(self, feedback_id):
        """
        Delete feedback.
        
        Parameters:
        -----------
        feedback_id : str
            ID of the feedback to delete
            
        Returns:
        --------
        dict
            Status of the deletion
        """
        try:
            # Load existing feedback
            df = pd.read_csv(self.storage_path)
            
            # Find the feedback
            mask = df['feedback_id'] == feedback_id
            if not mask.any():
                return {
                    'status': 'error',
                    'message': f'Feedback with ID {feedback_id} not found'
                }
            
            # Remove the feedback
            df = df[~mask]
            
            # Save updated dataframe
            df.to_csv(self.storage_path, index=False)
            
            return {
                'status': 'success',
                'message': f'Feedback {feedback_id} deleted successfully'
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error deleting feedback: {str(e)}'
            }
