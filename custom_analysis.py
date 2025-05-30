"""
Example script for analyzing custom student feedback data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import project modules
from src.preprocessing.text_processor import TextProcessor, extract_features
from src.models.supervised_models import SentimentClassifier, FeedbackCategorizer
from src.models.unsupervised_models import TopicModeler, FeedbackClusterer, DimensionalityReducer
from src.evaluation.metrics import evaluate_classification, evaluate_clustering
from src.visualization.visualizer import (
    plot_sentiment_distribution, plot_category_distribution,
    plot_wordcloud, plot_topic_wordcloud, plot_cluster_visualization
)

def analyze_custom_data(data_path, output_dir='results/custom'):
    """
    Analyze custom student feedback data.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing feedback data
    output_dir : str
        Directory to save results
    """
    print(f"Analyzing custom data from: {data_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    print("\nLoading data...")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} feedback samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check if required columns exist
    required_column = 'feedback_text'
    if required_column not in df.columns:
        print(f"Error: Required column '{required_column}' not found in the data")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Step 2: Preprocess text
    print("\nPreprocessing text...")
    text_processor = TextProcessor(
        remove_stopwords=True,
        remove_punctuation=True,
        lemmatize=True,
        stem=False,
        lowercase=True
    )
    df = text_processor.preprocess_dataframe(df, 'feedback_text')
    
    # Step 3: Extract features
    print("\nExtracting features...")
    feature_matrix, vectorizer = extract_features(
        df['processed_text'],
        method='tfidf',
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    # Step 4: Unsupervised learning
    
    # Topic modeling
    print("\nPerforming topic modeling...")
    topic_modeler = TopicModeler(method='lda', n_topics=5)
    topic_modeler.fit(df['processed_text'])
    
    top_words = topic_modeler.get_top_words_per_topic(n_words=10)
    print("\nTop words for each topic:")
    for i, words in enumerate(top_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Save topic word cloud
    fig = plot_topic_wordcloud(top_words)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topic_wordclouds.png")
    plt.close(fig)
    
    # Get topic distribution
    topic_distribution = topic_modeler.get_topic_distribution(df['processed_text'])
    df['dominant_topic'] = topic_distribution['Dominant_Topic']
    
    # Clustering
    print("\nPerforming clustering...")
    # Reduce dimensionality for clustering
    reducer = DimensionalityReducer(method='svd', n_components=50)
    reduced_features = reducer.fit_transform(feature_matrix)
    
    clusterer = FeedbackClusterer(method='kmeans', n_clusters=5)
    clusterer.fit(reduced_features)
    
    cluster_results = clusterer.evaluate(reduced_features)
    print(f"Silhouette Score: {cluster_results['silhouette_score']:.4f}")
    print("Cluster sizes:", cluster_results['cluster_sizes'])
    
    # Save cluster visualization
    cluster_labels = clusterer.model.labels_
    fig = plot_cluster_visualization(reduced_features, cluster_labels)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_visualization.png")
    plt.close(fig)
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Step 5: Supervised learning (if labels are available)
    if 'sentiment' in df.columns:
        print("\nPerforming sentiment analysis...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, df['sentiment'], test_size=0.3, random_state=42
        )
        
        # Train sentiment classifier
        sentiment_classifier = SentimentClassifier(model_type='logistic_regression')
        sentiment_classifier.train(X_train, y_train)
        
        # Evaluate
        sentiment_results = sentiment_classifier.evaluate(X_test, y_test)
        print(f"Sentiment Classification Accuracy: {sentiment_results['accuracy']:.4f}")
        print(f"Sentiment Classification F1 Score: {sentiment_results['f1_score']:.4f}")
    else:
        print("\nNo sentiment labels found. Skipping sentiment analysis.")
    
    if 'category' in df.columns:
        print("\nPerforming category classification...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, df['category'], test_size=0.3, random_state=42
        )
        
        # Train category classifier
        category_classifier = FeedbackCategorizer(model_type='random_forest')
        category_classifier.train(X_train, y_train)
        
        # Evaluate
        category_results = category_classifier.evaluate(X_test, y_test)
        print(f"Category Classification Accuracy: {category_results['accuracy']:.4f}")
        print(f"Category Classification F1 Score: {category_results['f1_score']:.4f}")
    else:
        print("\nNo category labels found. Skipping category classification.")
    
    # Step 6: Save results
    print(f"\nSaving results to {output_dir}...")
    
    # Save processed data
    df.to_csv(f"{output_dir}/processed_data.csv", index=False)
    
    # Generate and save word cloud
    fig = plot_wordcloud(df['processed_text'].str.cat(sep=' '))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feedback_wordcloud.png")
    plt.close(fig)
    
    # Save topic distribution
    topic_distribution.to_csv(f"{output_dir}/topic_distribution.csv", index=False)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze custom student feedback data')
    parser.add_argument('data_path', type=str, help='Path to the CSV file containing feedback data')
    parser.add_argument('--output', type=str, default='results/custom', help='Directory to save results')
    
    args = parser.parse_args()
    
    analyze_custom_data(args.data_path, args.output)
