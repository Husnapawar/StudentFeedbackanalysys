"""
Demo script to show how to use the Student Feedback Analysis System programmatically.
"""

import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.text_processor import TextProcessor, extract_features
from src.models.supervised_models import SentimentClassifier
from src.models.unsupervised_models import TopicModeler
from src.visualization.visualizer import plot_wordcloud, plot_topic_wordcloud

def demo():
    """Demonstrate how to use the system programmatically."""
    print("Student Feedback Analysis System - Demo")
    print("=" * 40)

    # Step 1: Load some example feedback
    example_feedback = [
        "The course content was excellent and very well organized.",
        "I found the teaching quality to be below average.",
        "The learning resources were adequate but could be improved.",
        "Great support services, they really helped me succeed.",
        "The assessment methods were fair and appropriate."
    ]

    df = pd.DataFrame({'feedback_text': example_feedback})
    print(f"\nLoaded {len(df)} example feedback items")

    # Step 2: Preprocess the text
    print("\nPreprocessing text...")
    text_processor = TextProcessor(
        remove_stopwords=True,
        remove_punctuation=True,
        lemmatize=True,
        stem=False,
        lowercase=True
    )

    df = text_processor.preprocess_dataframe(df, 'feedback_text')

    for i, (original, processed) in enumerate(zip(df['feedback_text'], df['processed_text'])):
        print(f"\nOriginal [{i+1}]: {original}")
        print(f"Processed [{i+1}]: {processed}")

    # Step 3: Extract features
    print("\nExtracting features...")
    feature_matrix, vectorizer = extract_features(
        df['processed_text'],
        method='tfidf',
        max_features=1000,
        ngram_range=(1, 2)
    )

    # Step 4: Sentiment analysis
    print("\nPerforming sentiment analysis...")
    # For demo purposes, we'll use a pre-trained model
    # In a real scenario, you would train this on labeled data
    sentiment_classifier = SentimentClassifier(model_type='logistic_regression')

    # Normally you would train the model first:
    # sentiment_classifier.train(X_train, y_train)

    # For demo, we'll just predict using the model's default parameters
    # This won't give accurate results without training, but demonstrates the API
    sentiments = ['Positive', 'Negative', 'Neutral', 'Positive', 'Neutral']
    print("Predicted sentiments (simulated):")
    for i, (text, sentiment) in enumerate(zip(example_feedback, sentiments)):
        print(f"[{i+1}] {sentiment}: {text[:50]}...")

    # Step 5: Topic modeling
    print("\nPerforming topic modeling...")
    topic_modeler = TopicModeler(method='lda', n_topics=2)
    topic_modeler.fit(df['processed_text'])

    top_words = topic_modeler.get_top_words_per_topic(n_words=5)
    print("\nTop words for each topic:")
    for i, words in enumerate(top_words):
        print(f"Topic {i+1}: {', '.join(words)}")

    # Step 6: Visualization
    print("\nGenerating visualizations...")

    # Word cloud
    wordcloud_fig = plot_wordcloud(df['processed_text'].str.cat(sep=' '))
    # The plot_wordcloud function already returns a figure, so we just save it
    wordcloud_fig.savefig('demo_wordcloud.png')

    # Topic word clouds
    topic_wordcloud_fig = plot_topic_wordcloud(top_words)
    # The plot_topic_wordcloud function already returns a figure, so we just save it
    topic_wordcloud_fig.savefig('demo_topic_wordclouds.png')

    print("\nDemo complete! Visualizations saved as demo_wordcloud.png and demo_topic_wordclouds.png")

if __name__ == "__main__":
    demo()
