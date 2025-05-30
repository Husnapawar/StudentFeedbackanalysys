"""
Main script for student feedback analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import project modules
from utils.data_generator import generate_sample_feedback
from preprocessing.text_processor import TextProcessor, extract_features
from models.supervised_models import SentimentClassifier, FeedbackCategorizer
from models.unsupervised_models import TopicModeler, FeedbackClusterer, DimensionalityReducer
from evaluation.metrics import evaluate_classification, evaluate_clustering, evaluate_topic_model
from visualization.visualizer import (
    plot_sentiment_distribution, plot_category_distribution,
    plot_wordcloud, plot_topic_wordcloud, plot_cluster_visualization,
    plot_confusion_matrix
)

def main():
    """
    Main function to demonstrate the student feedback analysis workflow.
    """
    print("Student Feedback Analysis System")
    print("=" * 40)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Generate or load sample data
    data_path = 'data/sample_feedback.csv'
    if not os.path.exists(data_path):
        print("\nGenerating sample feedback data...")
        df = generate_sample_feedback(n_samples=1000, output_path=data_path)
    else:
        print("\nLoading existing feedback data...")
        df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} feedback samples")
    print("\nSample feedback:")
    print(df[['feedback_text', 'rating', 'true_sentiment', 'true_category']].head())
    
    # Step 2: Preprocess the text data
    print("\nPreprocessing feedback text...")
    text_processor = TextProcessor(
        remove_stopwords=True,
        remove_punctuation=True,
        lemmatize=True,
        stem=False,
        lowercase=True
    )
    df = text_processor.preprocess_dataframe(df, 'feedback_text')
    
    print("\nSample processed text:")
    for i, (original, processed) in enumerate(zip(df['feedback_text'].head(3), df['processed_text'].head(3))):
        print(f"\nOriginal [{i+1}]: {original}")
        print(f"Processed [{i+1}]: {processed}")
    
    # Step 3: Extract features
    print("\nExtracting features...")
    feature_matrix, vectorizer = extract_features(
        df['processed_text'],
        method='tfidf',
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    feature_names = vectorizer.get_feature_names_out()
    print(f"Extracted {len(feature_names)} features")
    
    # Step 4: Split data for supervised learning
    print("\nSplitting data for supervised learning...")
    X_train, X_test, y_train_sentiment, y_test_sentiment = train_test_split(
        feature_matrix, df['true_sentiment'], test_size=0.3, random_state=42
    )
    
    _, _, y_train_category, y_test_category = train_test_split(
        feature_matrix, df['true_category'], test_size=0.3, random_state=42
    )
    
    # Step 5: Train and evaluate sentiment classifier
    print("\nTraining sentiment classifier...")
    sentiment_classifier = SentimentClassifier(model_type='logistic_regression')
    sentiment_classifier.train(X_train, y_train_sentiment)
    
    print("\nEvaluating sentiment classifier...")
    sentiment_results = sentiment_classifier.evaluate(X_test, y_test_sentiment)
    print(f"Accuracy: {sentiment_results['accuracy']:.4f}")
    print(f"F1 Score: {sentiment_results['f1_score']:.4f}")
    
    # Step 6: Train and evaluate feedback categorizer
    print("\nTraining feedback categorizer...")
    feedback_categorizer = FeedbackCategorizer(model_type='random_forest')
    feedback_categorizer.train(X_train, y_train_category)
    
    print("\nEvaluating feedback categorizer...")
    category_results = feedback_categorizer.evaluate(X_test, y_test_category)
    print(f"Accuracy: {category_results['accuracy']:.4f}")
    print(f"F1 Score: {category_results['f1_score']:.4f}")
    
    # Step 7: Topic modeling (unsupervised)
    print("\nPerforming topic modeling...")
    topic_modeler = TopicModeler(method='lda', n_topics=5)
    topic_modeler.fit(df['processed_text'])
    
    top_words = topic_modeler.get_top_words_per_topic(n_words=10)
    print("\nTop words for each topic:")
    for i, words in enumerate(top_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Step 8: Clustering (unsupervised)
    print("\nPerforming clustering...")
    # Reduce dimensionality for clustering
    reducer = DimensionalityReducer(method='svd', n_components=50)
    reduced_features = reducer.fit_transform(feature_matrix)
    
    clusterer = FeedbackClusterer(method='kmeans', n_clusters=5)
    clusterer.fit(reduced_features)
    
    cluster_results = clusterer.evaluate(reduced_features)
    print(f"Silhouette Score: {cluster_results['silhouette_score']:.4f}")
    print("Cluster sizes:", cluster_results['cluster_sizes'])
    
    # Step 9: Visualizations
    print("\nGenerating visualizations...")
    
    # Sentiment distribution
    fig1 = plot_sentiment_distribution(df['true_sentiment'])
    plt.tight_layout()
    plt.savefig('results/sentiment_distribution.png')
    
    # Category distribution
    fig2 = plot_category_distribution(df['true_category'])
    plt.tight_layout()
    plt.savefig('results/category_distribution.png')
    
    # Word cloud
    fig3 = plot_wordcloud(df['processed_text'].str.cat(sep=' '))
    plt.tight_layout()
    plt.savefig('results/feedback_wordcloud.png')
    
    # Topic word clouds
    fig4 = plot_topic_wordcloud(top_words)
    plt.tight_layout()
    plt.savefig('results/topic_wordclouds.png')
    
    # Cluster visualization
    cluster_labels = clusterer.model.labels_
    fig5 = plot_cluster_visualization(reduced_features, cluster_labels)
    plt.tight_layout()
    plt.savefig('results/cluster_visualization.png')
    
    # Confusion matrices
    y_pred_sentiment = sentiment_classifier.predict(X_test)
    cm_sentiment = evaluate_classification(y_test_sentiment, y_pred_sentiment)['confusion_matrix']
    fig6 = plot_confusion_matrix(cm_sentiment, class_names=np.unique(y_test_sentiment))
    plt.tight_layout()
    plt.savefig('results/sentiment_confusion_matrix.png')
    
    y_pred_category = feedback_categorizer.predict(X_test)
    cm_category = evaluate_classification(y_test_category, y_pred_category)['confusion_matrix']
    fig7 = plot_confusion_matrix(cm_category, class_names=np.unique(y_test_category))
    plt.tight_layout()
    plt.savefig('results/category_confusion_matrix.png')
    
    print("\nAnalysis complete! Results saved to 'results' directory.")

if __name__ == "__main__":
    main()
