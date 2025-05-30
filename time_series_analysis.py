"""
Time series analysis of student feedback.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def analyze_feedback_over_time(df, date_column='date', time_unit='month'):
    """
    Analyze feedback trends over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing feedback data
    date_column : str
        Name of the column containing dates
    time_unit : str
        Time unit for aggregation ('day', 'week', 'month', 'quarter', 'year')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated data over time
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df.set_index(date_column, inplace=True)
    
    # Resample by time unit
    if time_unit == 'day':
        grouped = df.resample('D')
    elif time_unit == 'week':
        grouped = df.resample('W')
    elif time_unit == 'month':
        grouped = df.resample('M')
    elif time_unit == 'quarter':
        grouped = df.resample('Q')
    elif time_unit == 'year':
        grouped = df.resample('Y')
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")
    
    # Aggregate data
    time_series = pd.DataFrame()
    
    # Count feedback by time period
    time_series['feedback_count'] = grouped.size()
    
    # Average rating by time period
    if 'rating' in df.columns:
        time_series['avg_rating'] = grouped['rating'].mean()
    
    # Sentiment counts by time period
    if 'true_sentiment' in df.columns:
        sentiment_counts = df.groupby([pd.Grouper(freq=time_unit.upper()[0]), 'true_sentiment']).size().unstack(fill_value=0)
        time_series = pd.concat([time_series, sentiment_counts], axis=1)
    
    # Category counts by time period
    if 'true_category' in df.columns:
        # Get top 3 categories
        top_categories = df['true_category'].value_counts().nlargest(3).index
        for category in top_categories:
            category_mask = df['true_category'] == category
            category_counts = df[category_mask].groupby(pd.Grouper(freq=time_unit.upper()[0])).size()
            time_series[f'category_{category.replace(" ", "_")}'] = category_counts
    
    return time_series

def plot_feedback_trends(time_series, output_dir='.'):
    """
    Plot feedback trends over time.
    
    Parameters:
    -----------
    time_series : pandas.DataFrame
        DataFrame with aggregated data over time
    output_dir : str
        Directory to save plots
    """
    # Plot feedback count over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_series.index, time_series['feedback_count'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Feedback Count')
    plt.title('Feedback Volume Over Time')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feedback_volume_trend.png')
    
    # Plot average rating over time
    if 'avg_rating' in time_series.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(time_series.index, time_series['avg_rating'], marker='o', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Average Rating')
        plt.title('Average Rating Over Time')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/average_rating_trend.png')
    
    # Plot sentiment trends
    sentiment_columns = [col for col in time_series.columns if col in ['Positive', 'Negative', 'Neutral']]
    if sentiment_columns:
        plt.figure(figsize=(12, 6))
        for sentiment in sentiment_columns:
            plt.plot(time_series.index, time_series[sentiment], marker='o', label=sentiment)
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Sentiment Trends Over Time')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_trends.png')
    
    # Plot category trends
    category_columns = [col for col in time_series.columns if col.startswith('category_')]
    if category_columns:
        plt.figure(figsize=(12, 6))
        for col in category_columns:
            category_name = col.replace('category_', '').replace('_', ' ')
            plt.plot(time_series.index, time_series[col], marker='o', label=category_name)
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Top Category Trends Over Time')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/category_trends.png')

def detect_anomalies(time_series, column='feedback_count', window=3, threshold=2):
    """
    Detect anomalies in time series data using rolling statistics.
    
    Parameters:
    -----------
    time_series : pandas.DataFrame
        DataFrame with aggregated data over time
    column : str
        Column to analyze for anomalies
    window : int
        Window size for rolling statistics
    threshold : float
        Number of standard deviations to consider as anomaly
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with anomaly indicators
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = time_series[column].rolling(window=window).mean()
    rolling_std = time_series[column].rolling(window=window).std()
    
    # Calculate upper and lower bounds
    upper_bound = rolling_mean + (rolling_std * threshold)
    lower_bound = rolling_mean - (rolling_std * threshold)
    
    # Identify anomalies
    anomalies = pd.DataFrame(index=time_series.index)
    anomalies[column] = time_series[column]
    anomalies['rolling_mean'] = rolling_mean
    anomalies['upper_bound'] = upper_bound
    anomalies['lower_bound'] = lower_bound
    anomalies['anomaly'] = (time_series[column] > upper_bound) | (time_series[column] < lower_bound)
    
    return anomalies

def plot_anomalies(anomalies, column='feedback_count', output_dir='.'):
    """
    Plot time series with anomalies highlighted.
    
    Parameters:
    -----------
    anomalies : pandas.DataFrame
        DataFrame with anomaly indicators
    column : str
        Column to plot
    output_dir : str
        Directory to save plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the time series
    plt.plot(anomalies.index, anomalies[column], label=column.replace('_', ' ').title())
    
    # Plot the rolling mean
    plt.plot(anomalies.index, anomalies['rolling_mean'], label='Rolling Mean', color='green', linestyle='--')
    
    # Plot the bounds
    plt.fill_between(
        anomalies.index,
        anomalies['upper_bound'],
        anomalies['lower_bound'],
        color='green',
        alpha=0.1,
        label='Normal Range'
    )
    
    # Highlight anomalies
    anomaly_points = anomalies[anomalies['anomaly']]
    plt.scatter(
        anomaly_points.index,
        anomaly_points[column],
        color='red',
        label='Anomalies',
        zorder=5
    )
    
    plt.xlabel('Date')
    plt.ylabel(column.replace('_', ' ').title())
    plt.title(f'Anomaly Detection in {column.replace("_", " ").title()}')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/anomaly_detection.png')

def main():
    """Main function to demonstrate time series analysis."""
    print("Time Series Analysis of Student Feedback")
    print("=" * 50)
    
    # Create output directory
    import os
    output_dir = 'time_series_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_path = 'data/sample_feedback.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} feedback samples")
    
    # Analyze feedback over time
    print("\nAnalyzing feedback trends over time...")
    time_series = analyze_feedback_over_time(df, time_unit='month')
    
    # Plot feedback trends
    print("\nPlotting feedback trends...")
    plot_feedback_trends(time_series, output_dir)
    
    # Detect anomalies
    print("\nDetecting anomalies in feedback volume...")
    anomalies = detect_anomalies(time_series, column='feedback_count', window=3, threshold=2)
    
    # Plot anomalies
    print("\nPlotting anomalies...")
    plot_anomalies(anomalies, output_dir=output_dir)
    
    # Save time series data
    time_series.to_csv(f'{output_dir}/feedback_time_series.csv')
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir} directory:")
    print("- feedback_volume_trend.png")
    print("- average_rating_trend.png")
    print("- sentiment_trends.png")
    print("- category_trends.png")
    print("- anomaly_detection.png")
    print("- feedback_time_series.csv")

if __name__ == "__main__":
    main()
