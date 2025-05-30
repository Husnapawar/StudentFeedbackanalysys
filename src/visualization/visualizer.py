"""
Visualization utilities for student feedback analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_sentiment_distribution(sentiments, figsize=(10, 6)):
    """
    Plot the distribution of sentiment labels.
    
    Parameters:
    -----------
    sentiments : array-like
        Array of sentiment labels
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count sentiment frequencies
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    # Create bar plot
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Sentiment Labels')
    
    # Add count labels on top of bars
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    return fig

def plot_category_distribution(categories, figsize=(12, 6)):
    """
    Plot the distribution of feedback categories.
    
    Parameters:
    -----------
    categories : array-like
        Array of category labels
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count category frequencies
    category_counts = pd.Series(categories).value_counts()
    
    # Create horizontal bar plot
    sns.barplot(y=category_counts.index, x=category_counts.values, ax=ax)
    
    # Add labels and title
    ax.set_ylabel('Category')
    ax.set_xlabel('Count')
    ax.set_title('Distribution of Feedback Categories')
    
    # Add count labels on bars
    for i, count in enumerate(category_counts.values):
        ax.text(count + 5, i, str(count), va='center')
    
    return fig

def plot_wordcloud(text_data, stopwords=None, figsize=(12, 8), max_words=200):
    """
    Generate a word cloud from text data.
    
    Parameters:
    -----------
    text_data : str or list
        Text data to visualize. If list, it will be joined with spaces.
    stopwords : set, optional
        Set of stopwords to exclude
    figsize : tuple
        Figure size
    max_words : int
        Maximum number of words to include
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Join text if it's a list
    if isinstance(text_data, list):
        text_data = ' '.join(text_data)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        max_words=max_words,
        stopwords=stopwords,
        background_color='white',
        collocations=False
    ).generate(text_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Feedback Text')
    
    return fig

def plot_topic_wordcloud(topic_words, topic_names=None, figsize=(15, 10), 
                         cols=2, max_words=50):
    """
    Generate word clouds for each topic.
    
    Parameters:
    -----------
    topic_words : list of list
        List of words for each topic
    topic_names : list, optional
        List of topic names
    figsize : tuple
        Figure size
    cols : int
        Number of columns in the subplot grid
    max_words : int
        Maximum number of words to include in each cloud
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    n_topics = len(topic_words)
    rows = int(np.ceil(n_topics / cols))
    
    if topic_names is None:
        topic_names = [f'Topic {i+1}' for i in range(n_topics)]
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (words, name) in enumerate(zip(topic_words, topic_names)):
        if i < len(axes):
            # Create word cloud for topic
            text = ' '.join(words)
            wordcloud = WordCloud(
                width=400, 
                height=200, 
                max_words=max_words,
                background_color='white',
                collocations=False
            ).generate(text)
            
            # Plot word cloud
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].axis('off')
            axes[i].set_title(name)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def plot_cluster_visualization(X, labels, method='tsne', figsize=(10, 8)):
    """
    Visualize clusters in 2D space.
    
    Parameters:
    -----------
    X : array-like
        Data features
    labels : array-like
        Cluster labels
    method : str
        Dimensionality reduction method ('tsne' or 'pca')
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Reduce dimensionality to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'tsne' or 'pca'.")
    
    # Transform data to 2D
    X_2d = reducer.fit_transform(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot clusters
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    # Add labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'Cluster Visualization using {method.upper()}')
    
    return fig

def plot_feature_importance(model, feature_names, top_n=20, figsize=(10, 8)):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot feature importances
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    return fig

def plot_rating_distribution(ratings, figsize=(10, 6)):
    """
    Plot the distribution of ratings.
    
    Parameters:
    -----------
    ratings : array-like
        Array of rating values
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count rating frequencies
    rating_counts = pd.Series(ratings).value_counts().sort_index()
    
    # Create bar plot
    sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Ratings')
    
    # Add count labels on top of bars
    for i, count in enumerate(rating_counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    return fig

def plot_confusion_matrix(cm, class_names=None, figsize=(10, 8), cmap='Blues'):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        List of class names
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]


    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    # Add labels and title
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')

    return fig
