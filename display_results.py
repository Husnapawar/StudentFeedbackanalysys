"""
Simple script to display the results of the student feedback analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_results():
    """Display the results of the student feedback analysis."""
    print("Student Feedback Analysis Results")
    print("=" * 40)
    
    # Load the sample data
    data_path = 'data/sample_feedback.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Display basic statistics
        print(f"\nTotal feedback samples: {len(df)}")
        
        # Sentiment distribution
        sentiment_counts = df['true_sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        # Category distribution
        category_counts = df['true_category'].value_counts()
        print("\nCategory Distribution:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
        
        # Rating distribution
        rating_counts = df['rating'].value_counts().sort_index()
        print("\nRating Distribution:")
        for rating, count in rating_counts.items():
            print(f"  {rating}: {count} ({count/len(df)*100:.1f}%)")
    
    # List available visualizations
    print("\nAvailable Visualizations:")
    for file in os.listdir('results'):
        if file.endswith('.png'):
            print(f"  - {file}")
    
    print("\nTo view these visualizations, you can use an image viewer or run:")
    print("python -c \"import matplotlib.pyplot as plt; import matplotlib.image as mpimg; img = mpimg.imread('results/FILE_NAME.png'); plt.figure(figsize=(10, 6)); plt.imshow(img); plt.axis('off'); plt.show()\"")

if __name__ == "__main__":
    display_results()
