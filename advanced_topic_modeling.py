"""
Advanced topic modeling for student feedback analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text for topic modeling."""
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def find_optimal_topics(texts, start=2, limit=10, step=1):
    """Find the optimal number of topics using coherence scores."""
    # Create dictionary and corpus
    processed_texts = [preprocess_text(text) for text in texts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Calculate coherence scores for different numbers of topics
    coherence_scores = []
    models_list = []
    topics_range = range(start, limit, step)
    
    for num_topics in topics_range:
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        models_list.append(lda_model)
        
        print(f'Number of Topics: {num_topics}, Coherence Score: {coherence_score:.4f}')
    
    # Find the model with the highest coherence score
    optimal_model_index = coherence_scores.index(max(coherence_scores))
    optimal_model = models_list[optimal_model_index]
    optimal_topics = topics_range[optimal_model_index]
    
    # Plot coherence scores
    plt.figure(figsize=(10, 6))
    plt.plot(topics_range, coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Topic Coherence Scores by Number of Topics')
    plt.xticks(topics_range)
    plt.grid(True)
    plt.savefig('topic_coherence_scores.png')
    
    return optimal_model, dictionary, corpus, optimal_topics

def visualize_topics(model, dictionary, corpus, num_topics):
    """Visualize topics using pyLDAvis."""
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
        
        # Prepare visualization
        vis_data = gensimvis.prepare(model, corpus, dictionary)
        
        # Save visualization to HTML file
        pyLDAvis.save_html(vis_data, 'topic_visualization.html')
        print("Interactive topic visualization saved to 'topic_visualization.html'")
        
    except ImportError:
        print("pyLDAvis not installed. Install with: pip install pyldavis")
        
        # Alternative: print top words for each topic
        print("\nTop words for each topic:")
        for topic_id in range(num_topics):
            top_words = [word for word, _ in model.show_topic(topic_id, topn=10)]
            print(f"Topic {topic_id+1}: {', '.join(top_words)}")

def main():
    """Main function to demonstrate advanced topic modeling."""
    print("Advanced Topic Modeling for Student Feedback")
    print("=" * 50)
    
    # Load data
    data_path = 'data/sample_feedback.csv'
    df = pd.read_csv(data_path)
    texts = df['feedback_text'].tolist()
    
    print(f"Loaded {len(texts)} feedback samples")
    
    # Find optimal number of topics
    print("\nFinding optimal number of topics...")
    optimal_model, dictionary, corpus, optimal_topics = find_optimal_topics(
        texts, start=2, limit=11, step=2
    )
    
    print(f"\nOptimal number of topics: {optimal_topics}")
    
    # Visualize topics
    print("\nVisualizing topics...")
    visualize_topics(optimal_model, dictionary, corpus, optimal_topics)
    
    # Analyze document-topic distribution
    print("\nAnalyzing document-topic distribution...")
    doc_topics = [optimal_model.get_document_topics(bow) for bow in corpus]
    
    # Get the dominant topic for each document
    dominant_topics = [max(topics, key=lambda x: x[1])[0] for topics in doc_topics]
    df['dominant_topic'] = [topic + 1 for topic in dominant_topics]  # 1-based indexing
    
    # Count documents per topic
    topic_counts = df['dominant_topic'].value_counts().sort_index()
    
    print("\nDocuments per topic:")
    for topic, count in topic_counts.items():
        print(f"Topic {topic}: {count} documents ({count/len(df)*100:.1f}%)")
    
    # Plot document distribution by topic
    plt.figure(figsize=(10, 6))
    plt.bar(topic_counts.index, topic_counts.values)
    plt.xlabel('Topic')
    plt.ylabel('Number of Documents')
    plt.title('Document Distribution by Topic')
    plt.xticks(topic_counts.index)
    plt.grid(axis='y')
    plt.savefig('document_topic_distribution.png')
    
    print("\nAnalysis complete!")
    print("Results saved to:")
    print("- topic_coherence_scores.png")
    print("- document_topic_distribution.png")
    print("- topic_visualization.html (if pyLDAvis is installed)")

if __name__ == "__main__":
    main()
