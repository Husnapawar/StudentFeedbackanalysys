"""
Quick demo script to show how to use the Student Feedback Analysis System programmatically.
"""

import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.text_processor import TextProcessor
from src.models.unsupervised_models import TopicModeler
from src.visualization.visualizer import plot_wordcloud

def quick_demo():
    """Run a quick demo of the system."""
    print("Student Feedback Analysis System - Quick Demo")
    print("=" * 50)
    
    # Step 1: Create some example feedback
    example_feedback = [
        "The course content was excellent and very well organized.",
        "I found the teaching quality to be below average.",
        "The learning resources were adequate but could be improved.",
        "Great support services, they really helped me succeed.",
        "The assessment methods were fair and appropriate."
    ]
    
    df = pd.DataFrame({'feedback_text': example_feedback})
    print(f"\nAnalyzing {len(df)} example feedback items:")
    for i, text in enumerate(example_feedback):
        print(f"[{i+1}] {text}")
    
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
    
    # Step 3: Topic modeling
    print("\nPerforming topic modeling...")
    topic_modeler = TopicModeler(method='lda', n_topics=2)
    topic_modeler.fit(df['processed_text'])
    
    top_words = topic_modeler.get_top_words_per_topic(n_words=5)
    print("\nTop words for each topic:")
    for i, words in enumerate(top_words):
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Step 4: Generate word cloud
    print("\nGenerating word cloud...")
    wordcloud_fig = plot_wordcloud(df['processed_text'].str.cat(sep=' '))
    wordcloud_fig.savefig('demo_wordcloud.png')
    
    print("\nDemo complete! Word cloud saved as demo_wordcloud.png")

if __name__ == "__main__":
    quick_demo()
