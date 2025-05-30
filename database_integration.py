"""
Database integration for student feedback analysis.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime

class FeedbackDatabase:
    """Class for managing feedback data in a SQLite database."""
    
    def __init__(self, db_path='feedback.db'):
        """
        Initialize the database connection.
        
        Parameters:
        -----------
        db_path : str
            Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Create the necessary tables if they don't exist."""
        # Create feedback table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            course_id TEXT,
            subject TEXT,
            feedback_text TEXT NOT NULL,
            rating INTEGER,
            date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create sentiment analysis table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER,
            sentiment TEXT,
            confidence REAL,
            model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (feedback_id) REFERENCES feedback (id)
        )
        ''')
        
        # Create category analysis table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS category_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER,
            category TEXT,
            confidence REAL,
            model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (feedback_id) REFERENCES feedback (id)
        )
        ''')
        
        # Create topic analysis table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER,
            topic_id INTEGER,
            probability REAL,
            model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (feedback_id) REFERENCES feedback (id)
        )
        ''')
        
        # Create topics table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT,
            top_words TEXT,
            model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
    
    def import_feedback(self, df):
        """
        Import feedback data from a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing feedback data
        
        Returns:
        --------
        int
            Number of records imported
        """
        required_columns = ['feedback_text']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Prepare data for insertion
        records = []
        for _, row in df.iterrows():
            record = (
                row.get('student_id', None),
                row.get('course_id', None),
                row.get('subject', None),
                row['feedback_text'],
                row.get('rating', None),
                row.get('date', datetime.now().strftime('%Y-%m-%d'))
            )
            records.append(record)
        
        # Insert data
        self.cursor.executemany('''
        INSERT INTO feedback (student_id, course_id, subject, feedback_text, rating, date)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', records)
        
        self.conn.commit()
        
        return len(records)
    
    def store_sentiment_analysis(self, results_df, model_name):
        """
        Store sentiment analysis results.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame containing sentiment analysis results
        model_name : str
            Name of the model used for analysis
            
        Returns:
        --------
        int
            Number of records stored
        """
        # Get feedback IDs
        feedback_texts = results_df['text'].tolist()
        placeholders = ','.join(['?'] * len(feedback_texts))
        
        self.cursor.execute(f'''
        SELECT id, feedback_text FROM feedback
        WHERE feedback_text IN ({placeholders})
        ''', feedback_texts)
        
        feedback_id_map = {row[1]: row[0] for row in self.cursor.fetchall()}
        
        # Prepare data for insertion
        records = []
        for _, row in results_df.iterrows():
            feedback_id = feedback_id_map.get(row['text'])
            if feedback_id:
                record = (
                    feedback_id,
                    row['predicted_sentiment'],
                    row.get('confidence', 0.0),
                    model_name
                )
                records.append(record)
        
        # Insert data
        self.cursor.executemany('''
        INSERT INTO sentiment_analysis (feedback_id, sentiment, confidence, model)
        VALUES (?, ?, ?, ?)
        ''', records)
        
        self.conn.commit()
        
        return len(records)
    
    def store_category_analysis(self, results_df, model_name):
        """
        Store category analysis results.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame containing category analysis results
        model_name : str
            Name of the model used for analysis
            
        Returns:
        --------
        int
            Number of records stored
        """
        # Get feedback IDs
        feedback_texts = results_df['text'].tolist()
        placeholders = ','.join(['?'] * len(feedback_texts))
        
        self.cursor.execute(f'''
        SELECT id, feedback_text FROM feedback
        WHERE feedback_text IN ({placeholders})
        ''', feedback_texts)
        
        feedback_id_map = {row[1]: row[0] for row in self.cursor.fetchall()}
        
        # Prepare data for insertion
        records = []
        for _, row in results_df.iterrows():
            feedback_id = feedback_id_map.get(row['text'])
            if feedback_id:
                record = (
                    feedback_id,
                    row['predicted_category'],
                    row.get('confidence', 0.0),
                    model_name
                )
                records.append(record)
        
        # Insert data
        self.cursor.executemany('''
        INSERT INTO category_analysis (feedback_id, category, confidence, model)
        VALUES (?, ?, ?, ?)
        ''', records)
        
        self.conn.commit()
        
        return len(records)
    
    def store_topics(self, topic_words, model_name):
        """
        Store topic model results.
        
        Parameters:
        -----------
        topic_words : list of list
            List of top words for each topic
        model_name : str
            Name of the model used for analysis
            
        Returns:
        --------
        list
            List of topic IDs
        """
        # Prepare data for insertion
        records = []
        for i, words in enumerate(topic_words):
            record = (
                f"Topic {i+1}",
                f"Automatically generated topic {i+1}",
                ', '.join(words),
                model_name
            )
            records.append(record)
        
        # Insert data
        topic_ids = []
        for record in records:
            self.cursor.execute('''
            INSERT INTO topics (name, description, top_words, model)
            VALUES (?, ?, ?, ?)
            ''', record)
            topic_ids.append(self.cursor.lastrowid)
        
        self.conn.commit()
        
        return topic_ids
    
    def store_topic_analysis(self, doc_topic_matrix, feedback_texts, topic_ids, model_name):
        """
        Store document-topic distribution.
        
        Parameters:
        -----------
        doc_topic_matrix : numpy.ndarray
            Document-topic probability matrix
        feedback_texts : list
            List of feedback texts
        topic_ids : list
            List of topic IDs
        model_name : str
            Name of the model used for analysis
            
        Returns:
        --------
        int
            Number of records stored
        """
        # Get feedback IDs
        placeholders = ','.join(['?'] * len(feedback_texts))
        
        self.cursor.execute(f'''
        SELECT id, feedback_text FROM feedback
        WHERE feedback_text IN ({placeholders})
        ''', feedback_texts)
        
        feedback_id_map = {row[1]: row[0] for row in self.cursor.fetchall()}
        
        # Prepare data for insertion
        records = []
        for i, text in enumerate(feedback_texts):
            feedback_id = feedback_id_map.get(text)
            if feedback_id:
                for j, topic_id in enumerate(topic_ids):
                    probability = doc_topic_matrix[i, j]
                    record = (
                        feedback_id,
                        topic_id,
                        float(probability),
                        model_name
                    )
                    records.append(record)
        
        # Insert data
        self.cursor.executemany('''
        INSERT INTO topic_analysis (feedback_id, topic_id, probability, model)
        VALUES (?, ?, ?, ?)
        ''', records)
        
        self.conn.commit()
        
        return len(records)
    
    def get_feedback(self, limit=100):
        """
        Get feedback data from the database.
        
        Parameters:
        -----------
        limit : int
            Maximum number of records to retrieve
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feedback data
        """
        query = f'''
        SELECT * FROM feedback
        ORDER BY date DESC
        LIMIT {limit}
        '''
        
        return pd.read_sql_query(query, self.conn)
    
    def get_sentiment_analysis(self, limit=100):
        """
        Get sentiment analysis results from the database.
        
        Parameters:
        -----------
        limit : int
            Maximum number of records to retrieve
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing sentiment analysis results
        """
        query = f'''
        SELECT f.id, f.feedback_text, f.rating, f.date, s.sentiment, s.confidence, s.model
        FROM feedback f
        JOIN sentiment_analysis s ON f.id = s.feedback_id
        ORDER BY s.created_at DESC
        LIMIT {limit}
        '''
        
        return pd.read_sql_query(query, self.conn)
    
    def get_category_analysis(self, limit=100):
        """
        Get category analysis results from the database.
        
        Parameters:
        -----------
        limit : int
            Maximum number of records to retrieve
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing category analysis results
        """
        query = f'''
        SELECT f.id, f.feedback_text, f.rating, f.date, c.category, c.confidence, c.model
        FROM feedback f
        JOIN category_analysis c ON f.id = c.feedback_id
        ORDER BY c.created_at DESC
        LIMIT {limit}
        '''
        
        return pd.read_sql_query(query, self.conn)
    
    def get_topic_analysis(self, limit=100):
        """
        Get topic analysis results from the database.
        
        Parameters:
        -----------
        limit : int
            Maximum number of records to retrieve
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing topic analysis results
        """
        query = f'''
        SELECT f.id, f.feedback_text, t.name as topic_name, t.top_words, ta.probability
        FROM feedback f
        JOIN topic_analysis ta ON f.id = ta.feedback_id
        JOIN topics t ON ta.topic_id = t.id
        ORDER BY ta.probability DESC
        LIMIT {limit}
        '''
        
        return pd.read_sql_query(query, self.conn)

def main():
    """Main function to demonstrate database integration."""
    print("Database Integration for Student Feedback Analysis")
    print("=" * 50)
    
    # Initialize database
    db_path = 'feedback.db'
    if os.path.exists(db_path):
        os.remove(db_path)  # Remove existing database for demo
    
    db = FeedbackDatabase(db_path)
    
    # Load sample data
    data_path = 'data/sample_feedback.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} feedback samples")
    
    # Import feedback data
    print("\nImporting feedback data...")
    num_imported = db.import_feedback(df)
    print(f"Imported {num_imported} feedback records")
    
    # Simulate sentiment analysis
    print("\nStoring sentiment analysis results...")
    sentiment_results = pd.DataFrame({
        'text': df['feedback_text'],
        'predicted_sentiment': df['true_sentiment'],  # Using true sentiment for demo
        'confidence': np.random.uniform(0.7, 1.0, len(df))
    })
    num_stored = db.store_sentiment_analysis(sentiment_results, 'demo_sentiment_model')
    print(f"Stored {num_stored} sentiment analysis records")
    
    # Simulate category analysis
    print("\nStoring category analysis results...")
    category_results = pd.DataFrame({
        'text': df['feedback_text'],
        'predicted_category': df['true_category'],  # Using true category for demo
        'confidence': np.random.uniform(0.7, 1.0, len(df))
    })
    num_stored = db.store_category_analysis(category_results, 'demo_category_model')
    print(f"Stored {num_stored} category analysis records")
    
    # Simulate topic modeling
    print("\nStoring topic model results...")
    topic_words = [
        ['course', 'content', 'excellent', 'organized', 'helpful'],
        ['teaching', 'quality', 'instructor', 'clear', 'engaging'],
        ['assessment', 'fair', 'difficult', 'exam', 'grading'],
        ['resources', 'textbook', 'materials', 'readings', 'online'],
        ['support', 'services', 'staff', 'helpful', 'responsive']
    ]
    topic_ids = db.store_topics(topic_words, 'demo_topic_model')
    print(f"Stored {len(topic_ids)} topics")
    
    # Simulate document-topic distribution
    print("\nStoring document-topic distribution...")
    num_docs = len(df)
    num_topics = len(topic_ids)
    doc_topic_matrix = np.random.dirichlet(np.ones(num_topics), size=num_docs)
    num_stored = db.store_topic_analysis(
        doc_topic_matrix,
        df['feedback_text'].tolist(),
        topic_ids,
        'demo_topic_model'
    )
    print(f"Stored {num_stored} document-topic probability records")
    
    # Retrieve and display data
    print("\nRetrieving data from database...")
    
    # Get feedback
    feedback_df = db.get_feedback(limit=5)
    print("\nSample feedback data:")
    print(feedback_df[['id', 'feedback_text', 'rating', 'date']].head())
    
    # Get sentiment analysis
    sentiment_df = db.get_sentiment_analysis(limit=5)
    print("\nSample sentiment analysis results:")
    print(sentiment_df[['id', 'feedback_text', 'sentiment', 'confidence']].head())
    
    # Get category analysis
    category_df = db.get_category_analysis(limit=5)
    print("\nSample category analysis results:")
    print(category_df[['id', 'feedback_text', 'category', 'confidence']].head())
    
    # Get topic analysis
    topic_df = db.get_topic_analysis(limit=5)
    print("\nSample topic analysis results:")
    print(topic_df[['id', 'feedback_text', 'topic_name', 'probability']].head())
    
    # Close database connection
    db.close()
    
    print(f"\nDatabase integration demo complete! Database saved to {db_path}")

if __name__ == "__main__":
    main()
