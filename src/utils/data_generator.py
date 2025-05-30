"""
Utility to generate synthetic student feedback data for testing and development.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_feedback(n_samples=1000, output_path=None):
    """
    Generate synthetic student feedback data.
    
    Parameters:
    -----------
    n_samples : int
        Number of feedback samples to generate
    output_path : str, optional
        Path to save the generated data. If None, returns the DataFrame
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing synthetic feedback data if output_path is None
    """
    # Define possible course subjects
    subjects = ['Mathematics', 'Computer Science', 'Physics', 'Chemistry', 
                'Biology', 'History', 'Literature', 'Economics', 'Psychology']
    
    # Define possible feedback categories
    categories = ['Course Content', 'Teaching Quality', 'Assessment Methods', 
                  'Learning Resources', 'Facilities', 'Support Services']
    
    # Define sentiment categories and their probabilities
    sentiments = ['Positive', 'Neutral', 'Negative']
    sentiment_probs = [0.6, 0.25, 0.15]  # More positive than negative feedback
    
    # Generate random dates within the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(n_samples)]
    
    # Generate synthetic feedback text based on sentiment and category
    positive_templates = [
        "I really enjoyed the {subject} course. The {category} was excellent.",
        "The {category} in the {subject} class exceeded my expectations.",
        "Great {category} in the {subject} course, very helpful for my learning.",
        "The instructor's {category} approach in {subject} was outstanding.",
        "I found the {category} in {subject} to be very well organized and useful."
    ]
    
    neutral_templates = [
        "The {category} in the {subject} course was adequate.",
        "I think the {category} in {subject} was average, nothing special.",
        "The {subject} course had reasonable {category}, but could be improved.",
        "The {category} in {subject} met the basic requirements.",
        "Neither good nor bad {category} in the {subject} course."
    ]
    
    negative_templates = [
        "I was disappointed with the {category} in the {subject} course.",
        "The {category} in {subject} needs significant improvement.",
        "Poor {category} made the {subject} course difficult to follow.",
        "I struggled with {subject} due to inadequate {category}.",
        "The {category} in the {subject} class was below standard."
    ]
    
    # Generate data
    data = []
    for i in range(n_samples):
        subject = random.choice(subjects)
        category = random.choice(categories)
        sentiment = random.choices(sentiments, weights=sentiment_probs)[0]
        
        if sentiment == 'Positive':
            feedback_text = random.choice(positive_templates).format(subject=subject, category=category)
            rating = random.randint(4, 5)
        elif sentiment == 'Neutral':
            feedback_text = random.choice(neutral_templates).format(subject=subject, category=category)
            rating = random.randint(3, 4)
        else:  # Negative
            feedback_text = random.choice(negative_templates).format(subject=subject, category=category)
            rating = random.randint(1, 3)
        
        student_id = f"S{random.randint(10000, 99999)}"
        course_id = f"{subject[:3].upper()}{random.randint(100, 999)}"
        
        data.append({
            'student_id': student_id,
            'course_id': course_id,
            'subject': subject,
            'feedback_text': feedback_text,
            'rating': rating,
            'date': dates[i].strftime('%Y-%m-%d'),
            'true_category': category,  # For supervised learning evaluation
            'true_sentiment': sentiment  # For supervised learning evaluation
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file if output_path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
        return None
    
    return df

if __name__ == "__main__":
    # Generate and save sample data when script is run directly
    generate_sample_feedback(n_samples=1000, output_path="../data/sample_feedback.csv")
