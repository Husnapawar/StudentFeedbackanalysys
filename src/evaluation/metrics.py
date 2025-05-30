"""
Evaluation metrics for student feedback analysis models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(y_true, y_pred, labels=None):
    """
    Evaluate classification model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        List of label names
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'classification_report': report,
        'confusion_matrix': cm
    }

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
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    return fig

def evaluate_clustering(X, labels):
    """
    Evaluate clustering model performance.
    
    Parameters:
    -----------
    X : array-like
        Data features
    labels : array-like
        Cluster labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Calculate metrics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Skip -1 labels (noise) for silhouette calculation
    if -1 in unique_labels and n_clusters > 1:
        mask = labels != -1
        silhouette = silhouette_score(X[mask], labels[mask])
    elif n_clusters > 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = None
    
    # Calculate other metrics if there are at least 2 clusters
    if n_clusters > 1:
        try:
            db_score = davies_bouldin_score(X, labels)
        except:
            db_score = None
        
        try:
            ch_score = calinski_harabasz_score(X, labels)
        except:
            ch_score = None
    else:
        db_score = None
        ch_score = None
    
    # Calculate cluster sizes
    cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
    
    return {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score,
        'cluster_sizes': cluster_sizes
    }

def evaluate_topic_model(model, texts, feature_names=None, n_top_words=10):
    """
    Evaluate topic model performance.
    
    Parameters:
    -----------
    model : object
        Fitted topic model (LDA or NMF)
    texts : list or pandas.Series
        Collection of preprocessed text documents
    feature_names : list, optional
        List of feature names
    n_top_words : int
        Number of top words to display per topic
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Get topic-word distributions
    topic_word_dist = model.components_
    
    # Get document-topic distributions
    if hasattr(model, 'transform'):
        doc_topic_dist = model.transform(texts)
    else:
        doc_topic_dist = None
    
    # Get top words per topic
    top_words_per_topic = []
    if feature_names is not None:
        for topic_idx, topic in enumerate(topic_word_dist):
            top_indices = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            top_words_per_topic.append(top_words)
    
    # Calculate topic coherence (if available)
    topic_coherence = None
    
    return {
        'top_words_per_topic': top_words_per_topic,
        'topic_word_dist': topic_word_dist,
        'doc_topic_dist': doc_topic_dist,
        'topic_coherence': topic_coherence
    }

def compare_models(model_results, metric_name, model_names=None):
    """
    Compare multiple models based on a specific metric.
    
    Parameters:
    -----------
    model_results : list of dict
        List of model evaluation results
    metric_name : str
        Name of the metric to compare
    model_names : list, optional
        List of model names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with model comparison
    """
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(model_results))]
    
    comparison = []
    for i, result in enumerate(model_results):
        if metric_name in result:
            comparison.append({
                'Model': model_names[i],
                metric_name: result[metric_name]
            })
    
    return pd.DataFrame(comparison)
