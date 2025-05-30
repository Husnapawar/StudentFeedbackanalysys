"""
Advanced sentiment analysis for student feedback using transformer models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def analyze_sentiment_with_transformers(texts, labels=None, test_size=0.3):
    """
    Analyze sentiment using transformer models.
    
    Parameters:
    -----------
    texts : list
        List of text documents
    labels : list, optional
        List of sentiment labels
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    dict
        Dictionary containing results
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import pipeline
        import torch
        
        print("Using transformers for sentiment analysis...")
        
        # Load pre-trained model and tokenizer
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        
        # Analyze sentiment
        results = []
        for text in texts:
            try:
                result = sentiment_analyzer(text)[0]
                # Convert to our format (Positive, Negative, Neutral)
                if result['label'] == 'POSITIVE':
                    sentiment = 'Positive'
                elif result['label'] == 'NEGATIVE':
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                results.append({
                    'text': text,
                    'predicted_sentiment': sentiment,
                    'confidence': result['score']
                })
            except Exception as e:
                print(f"Error analyzing text: {e}")
                results.append({
                    'text': text,
                    'predicted_sentiment': 'Neutral',
                    'confidence': 0.0
                })
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        # Evaluate if labels are provided
        if labels is not None:
            results_df['true_sentiment'] = labels
            
            # Calculate metrics
            accuracy = accuracy_score(results_df['true_sentiment'], results_df['predicted_sentiment'])
            report = classification_report(results_df['true_sentiment'], results_df['predicted_sentiment'], output_dict=True)
            
            # Create confusion matrix
            cm = confusion_matrix(results_df['true_sentiment'], results_df['predicted_sentiment'])
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=report.keys(), yticklabels=report.keys())
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('transformer_sentiment_confusion_matrix.png')
            
            # Return evaluation results
            return {
                'results_df': results_df,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm
            }
        
        return {'results_df': results_df}
    
    except ImportError:
        print("Transformers library not installed. Install with: pip install transformers torch")
        
        # Fallback to a simple rule-based approach
        print("Falling back to rule-based sentiment analysis...")
        
        positive_words = ['excellent', 'great', 'good', 'helpful', 'outstanding', 'enjoyed', 'useful']
        negative_words = ['poor', 'bad', 'disappointing', 'inadequate', 'below', 'difficult', 'struggled']
        
        results = []
        for text in texts:
            text_lower = text.lower()
            
            # Count positive and negative words
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            # Determine sentiment
            if pos_count > neg_count:
                sentiment = 'Positive'
                confidence = min(0.5 + 0.1 * (pos_count - neg_count), 0.9)
            elif neg_count > pos_count:
                sentiment = 'Negative'
                confidence = min(0.5 + 0.1 * (neg_count - pos_count), 0.9)
            else:
                sentiment = 'Neutral'
                confidence = 0.5
            
            results.append({
                'text': text,
                'predicted_sentiment': sentiment,
                'confidence': confidence
            })
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        # Evaluate if labels are provided
        if labels is not None:
            results_df['true_sentiment'] = labels
            
            # Calculate metrics
            accuracy = accuracy_score(results_df['true_sentiment'], results_df['predicted_sentiment'])
            report = classification_report(results_df['true_sentiment'], results_df['predicted_sentiment'], output_dict=True)
            
            # Create confusion matrix
            cm = confusion_matrix(results_df['true_sentiment'], results_df['predicted_sentiment'])
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=report.keys(), yticklabels=report.keys())
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix (Rule-based)')
            plt.tight_layout()
            plt.savefig('rule_based_sentiment_confusion_matrix.png')
            
            # Return evaluation results
            return {
                'results_df': results_df,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm
            }
        
        return {'results_df': results_df}

def main():
    """Main function to demonstrate advanced sentiment analysis."""
    print("Advanced Sentiment Analysis for Student Feedback")
    print("=" * 50)
    
    # Load data
    data_path = 'data/sample_feedback.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} feedback samples")
    
    # Split data for evaluation
    texts = df['feedback_text'].tolist()
    labels = df['true_sentiment'].tolist()
    
    # Analyze sentiment
    print("\nAnalyzing sentiment...")
    results = analyze_sentiment_with_transformers(texts, labels)
    
    # Print results
    if 'accuracy' in results:
        print(f"\nAccuracy: {results['accuracy']:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        for label, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"{label}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
    
    # Plot sentiment distribution
    results_df = results['results_df']
    sentiment_counts = results_df['predicted_sentiment'].value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Predicted Sentiment Distribution')
    for i, count in enumerate(sentiment_counts.values):
        plt.text(i, count + 5, str(count), ha='center')
    plt.tight_layout()
    plt.savefig('predicted_sentiment_distribution.png')
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['confidence'], bins=20, kde=True)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Sentiment Prediction Confidence Distribution')
    plt.tight_layout()
    plt.savefig('sentiment_confidence_distribution.png')
    
    # Save results to CSV
    results_df.to_csv('sentiment_analysis_results.csv', index=False)
    
    print("\nAnalysis complete!")
    print("Results saved to:")
    print("- sentiment_analysis_results.csv")
    print("- predicted_sentiment_distribution.png")
    print("- sentiment_confidence_distribution.png")
    if 'accuracy' in results:
        if 'transformers' in sys.modules:
            print("- transformer_sentiment_confusion_matrix.png")
        else:
            print("- rule_based_sentiment_confusion_matrix.png")

if __name__ == "__main__":
    import sys
    main()
